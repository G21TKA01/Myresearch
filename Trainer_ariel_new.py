import os
import sys
import math
import numpy as np
import cupy as cp
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
from utils.util import AverageMeter, ensure_dir
from tqdm import tqdm
from utils.metrics import Evaluator
import shutil
import utils.mask_gen as mask_gen
from torch.cuda.amp import autocast
from torch.cuda.amp import grad_scaler
from loss import CrossEntropy2d
import cv2
import csv
from calcurate_classmix import output_result
from natsort import natsorted
import glob
import shutil
from eval_stride_classmix import output_miou
from PIL import Image
from torchvision.io import read_image

STRIDE = 500
Threshold = 1000

rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=torch.device('cuda:0'))
rgb_std = torch.tensor([0.229, 0.224, 0.225], device=torch.device('cuda:0'))

mold_mapping = { 0:(0, 0, 0),
                    1:(255, 0, 255),
                    2:(255, 0, 0),
                    3:(255, 255, 0),
                    4:(0, 255, 0), }

mask_mapping = { 0:(0, 0, 0),
                    1:(255, 255, 255),}

def visualize(temp,map):
    r = temp.detach().clone()
    g = temp.detach().clone()
    b = temp.detach().clone()
    for l in range(0,len(map)):
        r[temp==l]=map[l][0]
        g[temp==l]=map[l][1]
        b[temp==l]=map[l][2]

    rgb = torch.zeros((temp.shape[1], temp.shape[2],3), device=torch.device('cuda:0'))

    rgb[:,:,0] = (r)
    rgb[:,:,1] = (g)
    rgb[:,:,2] = (b)
    #print(rgb)
    rgb = rgb.permute(2,0,1)
    return rgb

class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.NLLLoss()

    # `x`と`y`はそれぞれモデル出力の値とGTの値
    def forward(self, x, y, eps=1e-7):
        # (BATCH, C, H, W) から (C, BATCH * H * W) に並べ直す(
        #print(x.shape)
        #x = x.contiguous().view(-1,5)
        #print(x.shape)
        # 微小値を足してからlog
        z = torch.log(x + eps)
        #print(z - torch.log(x))
        #print(y)
        #y = y.contiguous().view(-1, 1)
        #print(y.shape)
        #_, y = torch.max(y, -1)
        #print(y)
        #print(x.shape)
        #print(y.shape)
        return self.loss_fn(z, y)

class Trainer(object):

    def __init__(self,
                 model,
                 config,
                 args,
                 train_data_loader,
                 valid_data_loader,
                 train_unsup_data_loader0,
                 train_unsup_data_loader1,
                 begin_time,
                 resume_file=None):

        print("     + Training Start ... ...")
        # for general
        self.config = config
        self.args = args
        self.device = (self._device(self.args.gpu))
        self.model = model.to(self.device)

        self.train_data_loader = train_data_loader
        self.valid_data_loder = valid_data_loader
        self.unsupervised_train_loader_0 = train_unsup_data_loader0
        self.unsupervised_train_loader_1 = train_unsup_data_loader1

        # for time
        self.begin_time = begin_time  # part of ckpt name
        self.save_period = self.config.save_period  # for save ckpt

        self.model_name = self.config.model_name

        if self.config.use_seed:
            self.checkpoint_dir = os.path.join(self.args.output, self.model_name,
                                               self.begin_time + '_seed' + str(self.config.random_seed))
            self.log_dir = os.path.join(self.args.output, self.model_name,
                                        self.begin_time + '_seed' + str(self.config.random_seed), 'log')
        else:
            self.checkpoint_dir = os.path.join(self.args.output, self.model_name,
                                               self.begin_time)
            self.log_dir = os.path.join(self.args.output, self.model_name,
                                        self.begin_time, 'log')

        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.log_dir)

        # output config to log file
        log_file_path = os.path.join(self.log_dir, self.model_name + '.txt')
        self.config.write_to_file(log_file_path)

        self.history = {
            'train': {
                'epoch': [],
                'loss': [],
                'acc': [],
                'miou': [],
                'prec': [],
                'recall': [],
                'f_score': [],
            },
            'valid': {
                'epoch': [],
                'loss': [],
                'acc': [],
                'miou': [],
                'prec': [],
                'recall': [],
                'f_score': [],
            }
        }
        # for optimize
        self.weight_init_algorithm = self.config.init_algorithm
        self.current_lr = self.config.init_lr

        # for train
        self.start_epoch = 0
        self.early_stop = self.config.early_stop  # early stop steps
        self.monitor_mode = self.config.monitor.split('/')[0]
        self.monitor_metric = self.config.monitor.split('/')[1]
        self.monitor_best = 0
        self.monitor_best_after_mix = 0
        self.best_epoch = -1
        self.best_epoch_after_mix = -1
        self.not_improved_count = 0
        self.monitor_iou = 0
        self.monitor_iou_after_mix = 0

        # resume file
        self.resume_file = resume_file
        self.resume_ = True if resume_file else False
        if self.resume_file is not None:
            with open(log_file_path, 'a') as f:
                f.write('\n')
                f.write('resume_file:' + resume_file + '\n')


        #self.loss = self._loss().to(self.device)
        self.optimizer_1 = self._optimizer(lr_algorithm=self.config.lr_algorithm)
        self.optimizer_2 = self._optimizer(lr_algorithm=self.config.lr_algorithm)

        # monitor init
        if self.monitor_mode != 'off':
            assert self.monitor_mode in ['min', 'max']
            self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf

        if self.config.use_one_cycle_lr:
            self.lr_scheduler_1 = self._lr_scheduler_onecycle(self.optimizer_1)
            self.lr_scheduler_2 = self._lr_scheduler_onecycle(self.optimizer_2)
        else:
            self.lr_scheduler_1 = self._lr_scheduler_lambda(self.optimizer_1, last_epoch=self.start_epoch - 1)
            self.lr_scheduler_2 = self._lr_scheduler_lambda(self.optimizer_2, last_epoch=self.start_epoch - 1)

        # Evaluator
        self.evaluator = Evaluator(self.config.nb_classes, self.device)

    def _device(self, gpu):
        if gpu == -1:
            device = torch.device('cpu')
            return device
        else:
            device = torch.device('cuda:{}'.format(gpu))
            return device

    def _optimizer(self, lr_algorithm):
        assert lr_algorithm in ['adam', 'adamw', 'sgd']
        if lr_algorithm == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                   lr=self.current_lr,
                                   betas=(0.9, 0.999),
                                   eps=1e-08,
                                   weight_decay=self.config.weight_decay,
                                   amsgrad=False
                                   )
        elif lr_algorithm == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                  lr=self.current_lr,
                                  momentum=self.config.momentum,
                                  dampening=0,
                                  weight_decay=self.config.weight_decay,
                                  nesterov=True)
        elif lr_algorithm == 'adamw':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.current_lr,
                                    betas=(0.9, 0.999),
                                    eps=1e-08,
                                    weight_decay=self.config.weight_decay,
                                    amsgrad=False
                                    )
        return optimizer

    def _loss(self):
        loss = nn.CrossEntropyLoss()
        return loss

    def _lr_scheduler_onecycle(self, optimizer):
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.config.init_lr * 6,
                                                     steps_per_epoch=len(self.train_data_loader),
                                                     epochs=self.config.epochs + 1,
                                                     div_factor=6)
        return lr_scheduler

    def _lr_scheduler_lambda(self, optimizer, last_epoch):
        lambda1 = lambda epoch: pow((1 - ((epoch - 1) / self.config.epochs)), 0.9)
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, last_epoch=last_epoch)

        return lr_scheduler

    def _weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find(' Conv') != -1:
            if self.weight_init_algorithm == 'kaiming':
                init.kaiming_normal_(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def train(self):
        if self.resume_ == False:
            # init weights
            self.model.apply(self._weight_init)
            print("     + Init weight ... Done !")
        else:
            # load the checkpoint file: resume_file
            self._resume_ckpt(resume_file=self.resume_file)
            print("     + Loading pth model file ... Done!")

        with open(self.log_dir + "/loss.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["Train_Loss", "Mix_Loss", "Mix_Count", "Test_Loss"])
        
        with open(self.log_dir + "/test_loss.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["miou", "test_loss"])

        epochs = self.config.epochs
        assert self.start_epoch < epochs

        for epoch in range(self.start_epoch, epochs + 1):
            train_log = self._train_epoch_classmix(epoch)
            eval_log = self._eval_epoch(epoch)

            # lr update
            if not self.config.use_one_cycle_lr:
                if self.lr_scheduler_1 is not None:
                    self.lr_scheduler_1.step(epoch)
                    for param_group in self.optimizer_1.param_groups:
                        self.current_lr = param_group['lr']
                if self.lr_scheduler_2 is not None:
                    self.lr_scheduler_2.step(epoch)
                    for param_group in self.optimizer_2.param_groups:
                        self.current_lr = param_group['lr']

            best = False
            best_after_mix = False 
            if self.monitor_mode != 'off':
                improved = (self.monitor_mode == 'min' and eval_log[
                    'val_' + self.monitor_metric] < self.monitor_best) or \
                           (self.monitor_mode == 'max' and eval_log['val_' + self.monitor_metric] > self.monitor_best)
                if improved:
                    self.monitor_best = eval_log['val_' + self.monitor_metric]
                    self.monitor_iou = eval_log['val_MIoU']
                    best = True
                    self.best_epoch = eval_log['epoch']
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1

                if epoch > self.config.warmup_period:
                    improved = (self.monitor_mode == 'min' and eval_log[
                        'val_' + self.monitor_metric] < self.monitor_best_after_mix) or \
                            (self.monitor_mode == 'max' and eval_log['val_' + self.monitor_metric] > self.monitor_best_after_mix)
                    if improved:
                        self.monitor_best_after_mix = eval_log['val_' + self.monitor_metric]
                        self.monitor_iou_after_mix = eval_log['val_MIoU']
                        best_after_mix = True
                        self.best_epoch_after_mix = eval_log['epoch']

                if self.not_improved_count > self.early_stop:
                    print("     + Validation Performance didn\'t improve for {} epochs."
                          "     + Training stop :/"
                          .format(self.not_improved_count))
                    break
            if epoch % self.save_period == 0 or best == True or best_after_mix == True:
                self._save_ckpt(epoch, best=best, best_after_mix = best_after_mix)

        # save history file
        print("     + Saving History ... ... ")
        hist_path = os.path.join(self.log_dir, 'history.txt')
        with open(hist_path, 'w') as f:
            f.write(str(self.history))

    def _train_epoch_classmix(self, epoch):
        ave_total_loss = AverageMeter()

        with open(self.log_dir + "/loss.csv", 'a') as f:
            writer = csv.writer(f)

            scaler = grad_scaler.GradScaler()
            self.evaluator.reset()

            # set model mode
            self.model.train()
            #print(self.model)


            train_dataloader = iter(self.train_data_loader)
            unsupervised_dataloader_0 = iter(self.unsupervised_train_loader_0)
            unsupervised_dataloader_1 = iter(self.unsupervised_train_loader_1)

            criterion =  CrossEntropyLoss2d()

            max_samples = max(len(self.train_data_loader), len(self.unsupervised_train_loader_0)) * self.config.batch_size
            niters_per_epoch = max_samples // self.config.batch_size
            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm(range(niters_per_epoch), file=sys.stdout, bar_format=bar_format)
            
            total_cps_loss = 0.0
            total_label_loss = 0.0
            classmix_counter = 0
            mix_switch = [False,False]
            store = 0


            for idx in pbar:
                #print(self.model.branch1)
                train_minibatch = train_dataloader.next()
                #print(train_minibatch[2])
                imgs = train_minibatch[0].to(self.device, non_blocking=True)
                #print(imgs.size())
                #print(imgs)
                gts = train_minibatch[1].to(self.device, non_blocking=True)
                assert not torch.isnan(imgs).any()
                assert not torch.isnan(gts).any()
                #print(gts.size())
                self.optimizer_1.zero_grad()
                self.optimizer_2.zero_grad()

            
                cps_loss = 0.0
                if epoch > self.config.warmup_period:
                    unsup_minibatch_0 = unsupervised_dataloader_0.next()
                    unsup_minibatch_1 = unsupervised_dataloader_1.next()
                    unsup_imgs_0 = unsup_minibatch_0[0].to(self.device, non_blocking=True)
                    unsup_imgs_1 = unsup_minibatch_1[0].to(self.device, non_blocking=True)
                    switch = False

                    with torch.no_grad():
                        # Estimate the pseudo-label with branch#1 & supervise branch#2
                        #logits_u0_tea_1_dict = self.model(unsup_imgs_0, step=1)
                        #logits_u0_tea_1 = logits_u0_tea_1_dict["out"]
                        logits_u0_tea_1 = self.model(unsup_imgs_0, step=1)
                        prob_u0_tea_1 = torch.sigmoid(logits_u0_tea_1).detach()

                        if self.config.use_mix:
                            #logits_u1_tea_1_dict = self.model(unsup_imgs_1, step=1)
                            #logits_u1_tea_1 = logits_u1_tea_1_dict["out"]
                            logits_u1_tea_1 = self.model(unsup_imgs_1, step=1)
                            prob_u1_tea_1 = torch.sigmoid(logits_u1_tea_1).detach()
                            name_0 = unsup_minibatch_0[1]
                            name_1 = unsup_minibatch_1[1]
                            #print(name_0[0][-7:-4])
                            image_0_num  = name_0[0][-7:-4]
                            image_1_num  = name_1[0][-7:-4]

                            if image_0_num == "1_2" or image_0_num == "2_1" or image_0_num == "2_2" or image_0_num == "2_3" or image_0_num == "3_2":
                                if image_1_num == "1_2" or image_1_num == "2_1" or image_1_num == "2_2" or image_1_num == "2_3" or image_1_num == "3_2":
                                    switch = True
                                else :
                                    switch = False        
                            elif image_0_num == image_1_num:
                                switch == True
                            else:
                                switch == False


                        else:
                            prob_u1_tea_1 = torch.zeros_like(prob_u0_tea_1)

                        # Estimate the pseudo-label with branch#2 & supervise branch#1
                        #logits_u0_tea_2_dict = self.model(unsup_imgs_0, step=2)
                        #logits_u0_tea_2 = logits_u0_tea_2_dict["out"]
                        logits_u0_tea_2= self.model(unsup_imgs_0, step=2)
                        prob_u0_tea_2 = torch.sigmoid(logits_u0_tea_2).detach()

                        if self.config.use_mix:
                            #logits_u1_tea_2_dict = self.model(unsup_imgs_1, step=2)
                            #logits_u1_tea_2 = logits_u1_tea_2_dict["out"]
                            logits_u1_tea_2 = self.model(unsup_imgs_1, step=2)
                            prob_u1_tea_2 = torch.sigmoid(logits_u1_tea_2).detach()
                        else:
                            prob_u1_tea_2 = torch.zeros_like(prob_u0_tea_2)

                        ps_u1_tea_label_1 = torch.argmax(prob_u1_tea_1, dim=1)

                    batch_mix_masks = torch.zeros_like(ps_u1_tea_label_1)
                    for img_i in range(unsup_imgs_0.shape[0]):
                        classes = torch.unique(ps_u1_tea_label_1[img_i], sorted=True)
                        nclasses = classes.shape[0]
                        if nclasses > 2:
                            classes = classes[torch.Tensor(
                                np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).long()]
                        elif nclasses == 2:
                            classes = classes[1].unsqueeze(0)
                        elif nclasses == 1:
                            continue
                        batch_mix_masks[img_i] = mask_gen.generate_class_mask(ps_u1_tea_label_1[img_i], classes)

                    store_mask = batch_mix_masks
                    batch_mix_masks = batch_mix_masks.unsqueeze(1)
                    unsup_imgs_mixed = unsup_imgs_0 * (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks

                    # Mix teacher predictions using same mask
                    # the mask pixels are either 1 or 0
                    prob_cons_tea_1 = prob_u0_tea_1 * (1 - batch_mix_masks) + prob_u1_tea_1 * batch_mix_masks
                    prob_cons_tea_2 = prob_u0_tea_2 * (1 - batch_mix_masks) + prob_u1_tea_2 * batch_mix_masks

                    #print(prob_cons_tea_1.shape)

                    ps_label_1 = torch.argmax(prob_cons_tea_1, dim=1)
                    #print(ps_label_1.shape)
                    ps_label_2 = torch.argmax(prob_cons_tea_2, dim=1)

                    for img_i in range(unsup_imgs_0.shape[0]):
                        classes = torch.unique(ps_label_1[img_i], sorted=True)
                        nclasses = classes.shape[0]
                        #print(nclasses)
                        if nclasses > 2:
                            mix_switch[img_i] = True
                        else:
                            mix_switch[img_i] = False

                    if store < 10 and mix_switch[0] == True and switch == True      :

                        name_0 = unsup_minibatch_0[1]
                        name_1 = unsup_minibatch_1[1]
                        name_0[0][96:-4]

                        ps_label_1_visualize = visualize(ps_label_1, mold_mapping)
                        ps_label_1_numpy = ps_label_1_visualize.cpu().numpy()
                        ps_label_1_save = ps_label_1_numpy.transpose(1,2,0)

                        store_mask_visualize = visualize(store_mask, mask_mapping)
                        batch_mix_masks_numpy = store_mask_visualize.cpu().numpy()
                        masks_save = batch_mix_masks_numpy.transpose(1,2,0)
    
                        unsup_imgs_mixed_store_mul = unsup_imgs_mixed.mul(rgb_std.view(3, 1, 1))
                        unsup_imgs_mixed_store_add = unsup_imgs_mixed_store_mul.add(rgb_mean.view(3, 1, 1)).detach().cpu().numpy()
                        unsup_imgs_mixed_store = unsup_imgs_mixed_store_add.reshape([3,1000,1000])
                        unsup_imgs_mixed_store = unsup_imgs_mixed_store*255
                        unsup_imgs_mixed_store = unsup_imgs_mixed_store.astype(np.uint8)
                        unsup_imgs_mixed_store = unsup_imgs_mixed_store.transpose(1,2,0)
                        unsup_imgs_mixed_store = unsup_imgs_mixed_store[:,:,::-1]
                        # name_0_0 = unsup_minibatch_0[1][0]
                        # name_1_0 = unsup_minibatch_1[1][0]
                        # name_0_1 = unsup_minibatch_0[1][1]
                        # name_1_1 = unsup_minibatch_1[1][1]
                        # ps_label_1_numpy = ps_label_1.cpu().numpy()
                        # ps_label_2_numpy = ps_label_2.cpu().numpy()
                        # batch_mix_masks_numpy = store_mask.cpu().numpy()

                        #print(batch_mix_masks_numpy.shape)
                        #print(batch_mix_masks_numpy)

                        cv2.imwrite(self.log_dir + "/" + "ep" + str(epoch) + "_" + str(name_0[0][96:-4]) + "+" + str(name_1[0][96:-4]) + "_mask.png" , masks_save)#batch_mix_masks_numpy)
                        cv2.imwrite(self.log_dir + "/" + "ep" + str(epoch) + "_" + str(name_0[0][96:-4]) + "+" + str(name_1[0][96:-4]) + "_1.png" , ps_label_1_save)#ps_label_1_numpy)
                        cv2.imwrite(self.log_dir + "/" + "ep" + str(epoch) + "_" + str(name_0[0][96:-4]) + "+" + str(name_1[0][96:-4]) + "_img.png" , unsup_imgs_mixed_store)
                        store += 1

                        # cv2.imwrite(self.log_dir + "/" + "ep" + str(epoch) + "_" + name_0_0[96:-4] + "+" + name_1_0[96:-4] + "_mask.png" , batch_mix_masks_numpy[0,:,:])
                        # #print(unsup_imgs_mixed_numpy[0,:,:])
                        # #cv2.imwrite(self.log_dir + "/" + "ep" + str(epoch) + "_" + name_0_1[96:-4] + "+" + name_1_1[96:-4] + ".jpg" , unsup_imgs_mixed_numpy[1,:,:])
                        # cv2.imwrite(self.log_dir + "/" + "ep" + str(epoch) + "_" + name_0_0[96:-4] + "+" + name_1_0[96:-4] + "_1.png" , ps_label_1_numpy[0,:,:])
                        # cv2.imwrite(self.log_dir + "/" + "ep" + str(epoch) + "_" + name_0_0[96:-4] + "+" + name_1_0[96:-4] + "_2.png" , ps_label_2_numpy[0,:,:])
                        # #cv2.imwrite(self.log_dir + "/" + "ep" + str(epoch) + "_" + name_0_1[96:-4] + "+" + name_1_1[96:-4] + ".png" , ps_label_1_numpy[1,:,:])
                        # store += 1
                    

                with autocast():
                    if epoch > self.config.warmup_period and mix_switch[0] == True:  # warmup
                        # Get student#1 prediction for mixed image
                        #logits_cons_stu_1_dict = self.model(unsup_imgs_mixed, step=1)
                        #logits_cons_stu_1 = logits_cons_stu_1_dict["out"]
                        logits_cons_stu_1 = self.model(unsup_imgs_mixed, step=1)
                        # Get student#2 prediction for mixed image
                        #logits_cons_stu_2_dict = self.model(unsup_imgs_mixed, step=2)
                        #logits_cons_stu_2 = logits_cons_stu_2_dict["out"]
                        logits_cons_stu_2 = self.model(unsup_imgs_mixed, step=2)

                        ps_label_1 = ps_label_1.long()
                        ps_label_2 = ps_label_2.long()

                        cps_loss = criterion(logits_cons_stu_1, ps_label_2) + criterion(logits_cons_stu_2, ps_label_1)
                        classmix_counter += 1

                    # empirically set coefficient to 1.0
                    cps_loss = cps_loss * 1

                    # sup loss
                    #sup_logits_l_dict = self.model(imgs, step=1)
                    #sup_logits_r_dict = self.model(imgs, step=2)
                    #sup_logits_l = sup_logits_l_dict["out"]
                    #sup_logits_r = sup_logits_r_dict["out"]
                    sup_logits_l = self.model(imgs, step=1)
                    sup_logits_r = self.model(imgs, step=2)
                    assert not torch.isnan(imgs).any()
                    assert not torch.isnan(gts).any()
                    assert not torch.isnan(sup_logits_l).any()
                    assert not torch.isnan(sup_logits_r).any()
                    #print(sup_logits_l.size())

                    gts = gts.long()

                    #print(gts)
                    #print(sup_logits_l)

                    #criterion = CrossEntropy2d(ignore_label=255).cuda()

                    loss_sup_l = criterion(sup_logits_l, gts)
                    loss_sup_r = criterion(sup_logits_r, gts)
                    assert not torch.isnan(loss_sup_l).any()
                    assert not torch.isnan(loss_sup_r).any()
                    #assert not torch.isnan(cps_loss).any()                

                    loss = loss_sup_l + loss_sup_r + cps_loss
                    if epoch > self.config.warmup_period and mix_switch[0] == True and switch == True:
                        total_cps_loss += cps_loss.item()
                    else:
                        total_cps_loss += cps_loss

                    total_label_loss = total_label_loss + loss_sup_l.item() + loss_sup_r.item()

                    #if idx % 500 == 0:
                    #    print("L_Loss:" + str(loss_sup_l) + " R_Loss:" + str(loss_sup_r) + " CPS:" + str(cps_loss))

                    #print("supl:" + str(loss_sup_l)  + ", supr:" + str(loss_sup_r) ", cps:" + str(cps_loss))

                    #print(loss)
                    
                    scaler.scale(loss).backward()
                    #print(self.model.branch1.state_dict().keys())
                    #print(self.model.branch1.backbone.layer1[0].conv1.weight.grad)

                    #torch.autograd.set_detect_anomaly(True)
                    scaler.step(self.optimizer_1)
                    scaler.step(self.optimizer_2)
                    scaler.update()

                pred = torch.argmax(sup_logits_l, dim=1)
                pred = pred.view(-1).long()
                
                label = gts.view(-1).long()

                # Add batch sample into evaluator
                self.evaluator.add_batch(label, pred)
                ave_total_loss.update(loss.item())

                if self.config.use_one_cycle_lr:
                    # lr update
                    if self.lr_scheduler_1 is not None:
                        self.lr_scheduler_1.step()
                        for param_group in self.optimizer_1.param_groups:
                            self.current_lr = param_group['lr']
                    if self.lr_scheduler_2 is not None:
                        self.lr_scheduler_2.step()
                        for param_group in self.optimizer_2.param_groups:
                            self.current_lr = param_group['lr']

            print("total_cps_loss:" + str(total_cps_loss) + "  total_label_loss:" + str(total_label_loss) + "  counter:" + str(classmix_counter))
            writer.writerow([total_label_loss, total_cps_loss, classmix_counter])

            acc = self.evaluator.Pixel_Accuracy().cpu().detach().numpy()
            miou = self.evaluator.Mean_Intersection_over_Union().cpu().detach().numpy()
            TP, FP, FN, TN = self.evaluator.get_base_value()
            iou = self.evaluator.get_iou().cpu().detach().numpy()
            prec = self.evaluator.Pixel_Precision_Class().cpu().detach().numpy()
            recall = self.evaluator.Pixel_Recall_Class().cpu().detach().numpy()
            f1_score = self.evaluator.Pixel_F1_score_Class().cpu().detach().numpy()

            #  train log and return
            self.history['train']['epoch'].append(epoch)
            self.history['train']['loss'].append(ave_total_loss.average())
            self.history['train']['acc'].append(acc.tolist())
            self.history['train']['miou'].append(miou.tolist())

            self.history['train']['prec'].append(prec[1])
            self.history['train']['recall'].append(recall[1])
            self.history['train']['f_score'].append(f1_score[1])

            if self.config.nb_classes == 2:
                miou = iou[1]

            return {
                'epoch': epoch,
                'loss': ave_total_loss.average(),
                'acc': acc,
                'miou': miou,
                'prec': prec[1],
                'recall': recall[1],
                'f_score': f1_score[1],
            }

    def _eval_epoch(self, epoch):
        ave_total_loss = AverageMeter()
        self.evaluator.reset()
        # set model mode
        self.model.eval()
        criterion =  CrossEntropyLoss2d()
        ver = self.config.data_num
        to_dir = self.log_dir + "/result_ep" + str(epoch)
        ensure_dir(to_dir)

        if ver == 1:
            name = ['20201210-2.8.10.14--4-6','20201210-2.8.10.14--4-7','20201210-2.8.10.14--4-8','20201210-2.8.10.14--4-9','20201210-2.8.10.14--4-10','20201210-2.8.10.14--4-11','20201210-2.8.10.14--4-12','20201210-2.8.10.14--4-13','20201210-2.8.10.14--4-14','20201210-2.8.10.14--4-15']
        elif ver == 2:
            name = ['20201210-2.8.10.14--4-1','20201210-2.8.10.14--4-2','20201210-2.8.10.14--4-3','20201210-2.8.10.14--4-4','20201210-2.8.10.14--4-5','20201210-2.8.10.14--4-11','20201210-2.8.10.14--4-12','20201210-2.8.10.14--4-13','20201210-2.8.10.14--4-14','20201210-2.8.10.14--4-15']
        elif ver == 3:
            name = ['20201210-2.8.10.14--4-1','20201210-2.8.10.14--4-2','20201210-2.8.10.14--4-3','20201210-2.8.10.14--4-4','20201210-2.8.10.14--4-5', '20201210-2.8.10.14--4-6','20201210-2.8.10.14--4-7','20201210-2.8.10.14--4-8','20201210-2.8.10.14--4-9','20201210-2.8.10.14--4-10']
        else:
            name = ['20201210-2.8.10.14--4-1','20201210-2.8.10.14--4-2','20201210-2.8.10.14--4-3','20201210-2.8.10.14--4-4','20201210-2.8.10.14--4-5', '20201210-2.8.10.14--4-6','20201210-2.8.10.14--4-7','20201210-2.8.10.14--4-8','20201210-2.8.10.14--4-9','20201210-2.8.10.14--4-10', '20201210-2.8.10.14--4-11','20201210-2.8.10.14--4-12','20201210-2.8.10.14--4-13','20201210-2.8.10.14--4-14','20201210-2.8.10.14--4-15']

        with torch.no_grad():
            counter = 0
            img_max = cp.full((1,5,3000,3000),-0.1)
            result = np.full((3000,3000,3),0)
            for steps, (imgs, gts, filename) in enumerate(self.valid_data_loder, start=1):
                imgs = imgs.to(self.device, non_blocking=True)
                gts = gts.to(self.device, non_blocking=True)
              
                # sup loss
                #sup_logits_l_dict = self.model(imgs, step=1)
                #sup_logits_l = sup_logits_l_dict["out"]
                sup_logits_l = self.model(imgs, step=1)
                gts = gts.long()
                loss = criterion(sup_logits_l, gts)

                #pred_reshape = torch.reshape(sup_logits_l, (2, 5, 1000, 1000)).cpu()
                #pred_numpy = sup_logits_l.detach().cpu().numpy()
                pred_numpy = cp.asarray(sup_logits_l.detach())
                #cp.set_printoptions(threshold=cp.inf)
                #print(pred_numpy.shape)

                #pred_numpy_3000 = cp.pad(pred_numpy, [(0,0),(0,0),(0,2000),(0,2000)]) 
                pred_numpy_3000 = cp.pad(pred_numpy, [(0,0),(0,0),(int(counter/5) * STRIDE , int(2000 - (int(counter/5)*STRIDE))),((counter % 5) * STRIDE, 2000 - (counter % 5) *STRIDE)] )
                #print(pred_numpy_3000.shape)
                #print("(" + str(int(counter/5) * STRIDE) + "," + str(int(2000 - (int(counter/5)*STRIDE))) +"), (" + str((counter % 5) * STRIDE) + "," + str(2000 - (counter % 5) *STRIDE) + ")" )
                
                counter = counter + 1

                #print(steps)
                #print(filename)
                #print((counter-1) % 5 *STRIDE)
                #print(int((counter-1)/5) *STRIDE)
                
                img_max = img_max + pred_numpy_3000

                # for j in range(pred_numpy.shape[3]):#y
                #     for i in range(pred_numpy.shape[2]):#x
                #         for k in range(pred_numpy.shape[1]):#channel		
                #             #average						
                #             img_max[j+( int((counter-1)/5) *STRIDE)][i+( (counter-1) % 5*STRIDE)][k] += pred_numpy[0][k][j][i]
                
                if counter == 25:
                    pred = cp.argmax(img_max, 1)
                    np.set_printoptions(threshold=np.inf)
                    #print(pred)
                    #print(pred.shape)
                    #print(dtype(pred))
                    # pred = cp.asnumpy(pred)
                    # for i in range(3000):
                    #     for j in range(3000):
                    #         #max_box = -0.1
                                        
                    #         if pred[0][i][j] == 0:
                    #             result[i][j] = 0,0,0
                    #         #     max_box = img_max[0][0][i][j]
                            
                    #         elif pred[0][i][j] == 1:
                    #             result[i][j] = 255,0,255
                    #         elif pred[0][i][j] == 2:
                    #             result[i][j] = 255,0,0
                    #         elif pred[0][i][j] == 3:
                    #             result[i][j] = 255,255,0
                    #         elif pred[0][i][j] == 4:
                    #             result[i][j] = 0,255,0
                            #if max_box < img_max[0][i][j]:
                            #     result[j][i] = 255,0,255
                            #     max_box = img_max[0][1][i][j]
                                
                            # if max_box < img_max[0][2][i][j]:
                            #     result[j][i] = 255,0,0
                            #     max_box = img_max[0][2][i][j]
                                
                            # if max_box < img_max[0][3][i][j]:
                            #     result[j][i] = 255,255,0
                            #     max_box = img_max[0][3][i][j]
                                
                            # if max_box < img_max[0][4][i][j]:
                            #     result[j][i] = 0,255,0
                            #     max_box = img_max[0][4][i][j]
                    
                    #write_file_name = name[int(steps/25 -1)] + "_prediction.png"
                    #pil_img = Image.fromarray(pred)
                    #pil_img.save(os.path.join(to_dir,write_file_name))
                    #cv2.imwrite(os.path.join(to_dir,write_file_name), result,[cv2.IMWRITE_PNG_COMPRESSION,9])
                    counter = 0
                    img_max = cp.full((1,5,3000,3000),-0.1)
                    #result = np.full((3000,3000,3),0)

                    #pred = torch.argmax(sup_logits_l, dim=1)
                    #pred = pred.view(-1).long()
                    #print(pred.shape)
                    #label = gts.view(-1).long()
                    #print(label.shape)

                         
                    if self.config.data_num == 1 or self.config.data_num == 2 or self.config.data_num == 3:
                        path_ano = natsorted(glob.glob(os.path.join("/home/taiga/ClassHyPer-master/Test_Gray/ver" + str(self.config.data_num) , '*.png')))
                    
                    else:
                        path_ano = natsorted(glob.glob(os.path.join("/home/taiga/ClassHyPer-master/Test_Gray" , '*.png')))

                    label = read_image(path_ano[int(steps/25)-1])
                    label = label.to("cuda")
                    #label = cp.array(Image.open((path_ano[int(steps/25)-1])))
                    #print(label.shape)
                    #print(label.dtype)
                    #pred = cp.reshape(pred,(3000,3000))
                    pred = cp.asnumpy(pred)
                    pred = torch.from_numpy(pred.astype(np.uint8)).clone()
                    pred = pred.to("cuda")
                    #print(path_ano[int(steps/25)-1])
                    # Add batch sample into evaluator

                    self.evaluator.add_batch(label, pred)

                # update ave metrics
                ave_total_loss.update(loss.item())
                
            # if self.config.data_num == 1 or self.config.data_num == 2 or self.config.data_num == 3:
            #     path_ano = natsorted(glob.glob(os.path.join("/home/taiga/ClassHyPer-master/Test_Gray/ver" + str(self.config.data_num) , '*.png')))
              
            # else:
            #     path_ano = natsorted(glob.glob(os.path.join("/home/taiga/ClassHyPer-master/Test_Gray" , '*.png')))
               
            # path_res = natsorted(glob.glob(os.path.join(self.log_dir + "/result_ep" + str(epoch), '*.png')))
            
            # for x in range(len(path_ano)):
            #     result = cv2.imread(path_res[x])
            #     annotation = cv2.imread(path_ano[x])
            #     result_numpy = torch.from_numpy(result.astype(np.float32)).clone()
            #     annotation_numpy = torch.from_numpy(annotation.astype(np.float32)).clone()
            #     self.evaluator.add_batch(annotation_numpy, result_numpy)

                

            # calculate metrics
            acc = self.evaluator.Pixel_Accuracy().cpu().detach().numpy()
            miou = self.evaluator.Mean_Intersection_over_Union().cpu().detach().numpy()
            #miou = output_miou(path_ano,path_res)
            print(miou)
            #print(miou.shape)
            TP, FP, FN, TN = self.evaluator.get_base_value()
            iou = self.evaluator.get_iou().cpu().detach().numpy()
            prec = self.evaluator.Pixel_Precision_Class().cpu().detach().numpy()
            recall = self.evaluator.Pixel_Recall_Class().cpu().detach().numpy()
            f1_score = self.evaluator.Pixel_F1_score_Class().cpu().detach().numpy()

            print('Epoch {} validation done !'.format(epoch))
            print('lr: {:.8f}\n'
                  'MIoU: {:6.4f},       Accuracy: {:6.4f},    Loss: {:.6f},\n'
                  'Precision: {:6.4f},  Recall: {:6.4f},      F_Score: {:6.4f}'
                  .format(self.current_lr,
                          miou, acc, ave_total_loss.average(),
                          prec[1], recall[1], f1_score[1]))

        self.history['valid']['epoch'].append(epoch)
        self.history['valid']['loss'].append(ave_total_loss.average())
        self.history['valid']['acc'].append(acc.tolist())
        self.history['valid']['miou'].append(miou)
        self.history['valid']['prec'].append(prec[1])
        self.history['valid']['recall'].append(recall[1])
        self.history['valid']['f_score'].append(f1_score[1])


        if self.config.nb_classes == 2:
            miou = iou[1]

        with open(self.log_dir + "/test_loss.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow([miou, ave_total_loss.average()])


        #  validation log and return
        return {
            'epoch': epoch,
            'val_Loss': ave_total_loss.average(),
            'val_Accuracy': acc,
            'val_MIoU': miou,
            'val_Precision': prec[1],
            'val_Recall': recall[1],
            'val_F_score': f1_score[1],

        }

    def _save_ckpt(self, epoch, best, best_after_mix):
        # save model ckpt
        state = {
            'epoch': epoch,
            'arch': str(self.model),
            'history': self.history,
            'state_dict': self.model.state_dict(),
            'monitor_best': self.monitor_best,
            'monitor_best_after_mix' : self.monitor_best_after_mix,
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-ep{}.pth'.format(epoch))
        best_filename = os.path.join(self.checkpoint_dir, 'checkpoint-best.pth')
        best_after_mix_filename = os.path.join(self.checkpoint_dir, 'checkpoint-best_after_mix.pth')
        last_best_filename = os.path.join(self.checkpoint_dir,
                                          'checkpoint-ep{}-iou{:.4f}.pth'.format(epoch, self.monitor_iou))
        last_best_after_mix_filename = os.path.join(self.checkpoint_dir,
                                          'checkpoint_after_mix-ep{}-iou{:.4f}.pth'.format(epoch, self.monitor_iou))
        if best:
            # copy the last best model
            if os.path.exists(best_filename):
                shutil.copyfile(best_filename, last_best_filename)
            print("     + Saving Best Checkpoint : Epoch {}  path: {} ...  ".format(epoch, best_filename))
            torch.save(state, best_filename)
        else:
            start_save_epochs = 1
            if epoch > start_save_epochs:
                print("     + After {} epochs, saving Checkpoint per {} epochs, path: {} ... ".format(start_save_epochs,
                                                                                                      self.save_period,
                                                                                                      filename))
                torch.save(state, filename)
        if best_after_mix:
            # copy the last best model
            if os.path.exists(best_after_mix_filename):
                shutil.copyfile(best_after_mix_filename, last_best_after_mix_filename)
            print("     + Saving Best Checkpoint : Epoch {}  path: {} ...  ".format(epoch, best_after_mix_filename))
            torch.save(state, best_after_mix_filename)

    def _resume_ckpt(self, resume_file):
        resume_path = os.path.join(resume_file)

        print("     + Loading Checkpoint: {} ... ".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.monitor_best = 0.0

        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("     + Model State Loaded ! :D ")
        print("     + Optimizer State Loaded ! :D ")
        print("     + Checkpoint file: '{}' , Start epoch {} Loaded !\n"
              "     + Prepare to run ! ! !"
              .format(resume_path, self.start_epoch))
