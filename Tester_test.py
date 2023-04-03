import torch
import numpy as np
import cupy as cp
import os
import time
import torch.nn as nn
from tqdm import tqdm
from utils.util import AverageMeter, ensure_dir
from utils.metrics import Evaluator
import cv2
from torchvision.io import write_jpeg, write_png
from calcurate_classmix import output_result
from natsort import natsorted
import glob
import shutil
from eval_stride_classmix import output_miou
from PIL import Image
from torchvision.io import read_image



STRIDE = 500

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

class Tester(object):
    def __init__(self,
                 model,
                 config,
                 args,
                 test_data_loader,
                 class_name,
                 begin_time,
                 resume_file):

        # for general
        self.config = config
        self.args = args
        self.device = torch.device('cpu') if self.args.gpu == -1 else torch.device('cuda:{}'.format(self.args.gpu))
        self.class_name = class_name
        # for Test
        self.model = model.to(self.device)
        self.models = []

        self.loss = self._loss().to(self.device)

        # for time
        self.begin_time = begin_time

        # for data
        self.test_data_loader = test_data_loader

        # for resume/save path
        self.history = {
            "eval": {
                "loss": [],
                "acc": [],
                "miou": [],
                "time": [],
                "prec": [],
                "recall": [],
                "f_score": [],
            },
        }

        self.model_name = self.config.model_name

        # loading args.weight or the checkpoint-best.pth
        self.test_log_path = os.path.join(self.args.output, 'test', 'log', self.model_name,
                                          self.begin_time)
        ensure_dir(self.test_log_path)

        if self.config.use_seed:
            self.resume_ckpt_path = resume_file if resume_file is not None else \
                os.path.join(self.config.save_dir, self.model_name,
                             self.begin_time + '_seed' + str(self.config.random_seed), 'checkpoint-best.pth')
        else:
            self.resume_ckpt_path = resume_file if resume_file is not None else \
                os.path.join(self.config.save_dir, self.model_name,
                             self.begin_time, 'checkpoint-best.pth')

        self.evaluator = Evaluator(self.config.nb_classes, self.device)

    def _loss(self):
        loss = nn.CrossEntropyLoss()
        return loss

    def eval_and_predict(self):
        self._resume_ckpt()

        self.model.eval()
        self.evaluator.reset()

        ave_total_loss = AverageMeter()

        to_dir = "/home/taiga/ClassHyPer-master/result/" + "resize-true-data1"
        ensure_dir(to_dir)
        ver = self.config.data_num

        if ver == 1:
            name = ['20201210-2.8.10.14--4-6','20201210-2.8.10.14--4-7','20201210-2.8.10.14--4-8','20201210-2.8.10.14--4-9','20201210-2.8.10.14--4-10','20201210-2.8.10.14--4-11','20201210-2.8.10.14--4-12','20201210-2.8.10.14--4-13','20201210-2.8.10.14--4-14','20201210-2.8.10.14--4-15']
            #name = ['20201210-2.8.10.14--4-11','20201210-2.8.10.14--4-12','20201210-2.8.10.14--4-13','20201210-2.8.10.14--4-14','20201210-2.8.10.14--4-15']
        elif ver == 2:
            name = ['20201210-2.8.10.14--4-1','20201210-2.8.10.14--4-2','20201210-2.8.10.14--4-3','20201210-2.8.10.14--4-4','20201210-2.8.10.14--4-5','20201210-2.8.10.14--4-11','20201210-2.8.10.14--4-12','20201210-2.8.10.14--4-13','20201210-2.8.10.14--4-14','20201210-2.8.10.14--4-15']
        elif ver == 3:
            name = ['20201210-2.8.10.14--4-1','20201210-2.8.10.14--4-2','20201210-2.8.10.14--4-3','20201210-2.8.10.14--4-4','20201210-2.8.10.14--4-5', '20201210-2.8.10.14--4-6','20201210-2.8.10.14--4-7','20201210-2.8.10.14--4-8','20201210-2.8.10.14--4-9','20201210-2.8.10.14--4-10']
        else:
            name = ['20201210-2.8.10.14--4-1','20201210-2.8.10.14--4-2','20201210-2.8.10.14--4-3','20201210-2.8.10.14--4-4','20201210-2.8.10.14--4-5', '20201210-2.8.10.14--4-6','20201210-2.8.10.14--4-7','20201210-2.8.10.14--4-8','20201210-2.8.10.14--4-9','20201210-2.8.10.14--4-10', '20201210-2.8.10.14--4-11','20201210-2.8.10.14--4-12','20201210-2.8.10.14--4-13','20201210-2.8.10.14--4-14','20201210-2.8.10.14--4-15']

        with torch.no_grad():
            counter = 0
            criterion = CrossEntropyLoss2d
            img_max = cp.full((1,5,3000,3000),-0.0001)
            result = np.full((3000,3000,3),0)
            for steps, (imgs, gts, filename) in enumerate(self.test_data_loader, start=1):
                imgs = imgs.to(self.device, non_blocking=True)
                gts = gts.to(self.device, non_blocking=True)
              
                # sup loss
                sup_logits_l = self.model(imgs, step=1)
                #sup_logits_l = sup_logits_l_dict["out"]
                #sup_logits_l_dict = self.model(imgs, step=1)
                #sup_logits_l = sup_logits_l_dict["out"]

                gts = gts.long()
                #loss = criterion(sup_logits_l, gts)

                #pred_reshape = torch.reshape(sup_logits_l, (2, 5, 1000, 1000)).cpu()
                #pred_numpy = sup_logits_l.detach().cpu().numpy()
                pred_numpy = cp.asarray(sup_logits_l.detach())
                #cp.set_printoptions(threshold=cp.inf)
                #print(pred_numpy.shape)

                #pred_numpy_3000 = cp.pad(pred_numpy, [(0,0),(0,0),(0,2000),(0,2000)]) 
                pred_numpy_3000 = cp.pad(pred_numpy, [(0,0),(0,0),(int(counter/5) * STRIDE , int(2000 - (int(counter/5)*STRIDE))),((counter % 5) * STRIDE, 2000 - (counter % 5) *STRIDE)] )
 
                
                counter = counter + 1
                
                img_max = img_max + pred_numpy_3000

                #print(img_max)

                
                if counter == 25:
                    pred = cp.argmax(img_max, 1)
                    np.set_printoptions(threshold=np.inf)
                    write_file_name = name[int(steps/25 -1)] + "_prediction.png"
                    pred = cp.asnumpy(pred)
                    pred = np.reshape(pred,(3000,3000))
                    print(pred.shape)

                    for i in range(3000):
                        for j in range(3000):
                            max_box = -0.1
                            
                            if pred[j][i] == 0:
                                result[j][i] = 0,0,0
                                
                            if pred[j][i] == 1:
                                result[j][i] = 255,0,255
                                
                            if pred[j][i] == 2:
                                result[j][i] = 255,0,0
                    
                            if pred[j][i] == 3:
                                result[j][i] = 255,255,0
                       
                            if pred[j][i] == 4:
                                result[j][i] = 0,255,0


                    #pred = torch.from_numpy(pred.astype(np.uint8)).clone()
                    #pred = pred.to("cuda")
                    #pil_img = Image.fromarray(pred)
                    #pil_img.save(os.path.join(to_dir,write_file_name))
                    cv2.imwrite(os.path.join(to_dir,write_file_name), result,[cv2.IMWRITE_PNG_COMPRESSION])
                    counter = 0
                    img_max = cp.full((1,5,3000,3000),-0.1)
                    #result = np.full((3000,3000,3),0)

                    #pred = torch.argmax(sup_logits_l, dim=1)
                    #pred = pred.view(-1).long()
                    #print(pred.shape)
                    #label = gts.view(-1).long()
                    #print(label.shape)

                         
                    # if self.config.data_num == 1 or self.config.data_num == 2 or self.config.data_num == 3:
                    #     path_ano = natsorted(glob.glob(os.path.join("/home/taiga/ClassHyPer-master/Test_Gray/ver" + str(self.config.data_num) , '*.png')))
                    
                    # else:
                    #     path_ano = natsorted(glob.glob(os.path.join("/home/taiga/ClassHyPer-master/Test_Gray" , '*.png')))

                    # label = read_image(path_ano[int(steps/25)-1])
                    # label = label.to("cuda")
                    #label = cp.array(Image.open((path_ano[int(steps/25)-1])))
                    #print(label.shape)
                    #print(label.dtype)
                    #pred = cp.reshape(pred,(3000,3000))
                    # pred = cp.asnumpy(pred)
                    # pred = torch.from_numpy(pred.astype(np.uint8)).clone()
                    # pred = pred.to("cuda")
                    #print(path_ano[int(steps/25)-1])
                    # Add batch sample into evaluator

                    #print(label.shape)
                    #print(pred.shape)
                    #self.evaluator.add_batch(label, pred)

                # update ave metrics
                #ave_total_loss.update(loss.item())
                
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

        #     # display evaluation result
        #     print('Evaluation phase !\n'
        #           'Accuracy: {:6.4f}, Loss: {:.6f}'.format(
        #         acc, ave_total_loss.average()))
        #     np.set_printoptions(formatter={'int': '{: 9}'.format})
        #     print('Class:    ', self.class_name, ' Average')
        #     np.set_printoptions(formatter={'float': '{: 6.6f}'.format})
        #     print('IoU:      ', np.hstack((iou, np.average(iou))))
        #     print('Precision:', np.hstack((prec, np.average(prec))))
        #     print('Recall:   ', np.hstack((recall, np.average(recall))))
        #     print('F_Score:  ', np.hstack((f1_score, np.average(f1_score))))
        #     np.set_printoptions(formatter={'int': '{:14}'.format})
        #     print('Confusion_matrix:')
        #     print(confusion_matrix)

        #     print('Prediction Phase !\n'
        #           'Total Time cost: {:.2f}s\n'
        #           .format(total_time,
        #                   ))
        # self.history["eval"]["loss"].append(ave_total_loss.average())
        # self.history["eval"]["acc"].append(acc.tolist())
        # self.history["eval"]["miou"].append(iou.tolist())
        # self.history["eval"]["time"].append(total_time)

        # self.history["eval"]["prec"].append(prec.tolist())
        # self.history["eval"]["recall"].append(recall.tolist())
        # self.history["eval"]["f_score"].append(f1_score.tolist())

        # Save results to log file
        # print("     + Saved history of evaluation phase !")
        # hist_path = os.path.join(self.test_log_path, "test-result.txt")
        # with open(hist_path, 'w') as f:
        #     f.write(str(self.history).replace("'", '"'))
        #     f.write('\nConfusion_matrix:\n')
        #     f.write(str(confusion_matrix))

        #     np.set_printoptions(formatter={'int': '{: 9}'.format})
        #     f.write('\nClass:    ' + str(self.class_name) + '  Average')
        #     np.set_printoptions(formatter={'float': '{: 6.6f}'.format})
        #     format_iou = np.hstack((iou, np.average(iou)))
        #     format_prec = np.hstack((prec, np.average(prec)))
        #     format_recall = np.hstack((recall, np.average(recall)))
        #     format_f1_score = np.hstack((f1_score, np.average(f1_score)))
        #     f.write('\nIoU:      ' + str(format_iou))
        #     f.write('\nPrecision:' + str(format_prec))
        #     f.write('\nRecall:   ' + str(format_recall))
        #     f.write('\nF1_score: ' + str(format_f1_score))

    def _resume_ckpt(self):
        print("     + Loading ckpt path : {} ...".format(self.resume_ckpt_path))
        checkpoint = torch.load(self.resume_ckpt_path)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("     + Model State Loaded ! :D ")
        print("     + Checkpoint file: '{}' , Loaded ! \n"
              "     + Prepare to test ! ! !"
              .format(self.resume_ckpt_path))
