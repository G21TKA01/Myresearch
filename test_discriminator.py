from pathlib import Path
import os
import datetime 
import csv
import glob
#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
#from IPython import display
from PIL import Image
from torchvision.io import read_image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
#import torchvision.datasets as datasets
from models.vision_main.torchvision.datasets import folder as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torcheval.metrics import MulticlassAccuracy
from torcheval.metrics import MulticlassPrecision
from torcheval.metrics import MulticlassRecall

#save_dir = Path("/data")  # 適宜変更してください
dataset_dir = "/home/taiga/ClassHyPer-master/discri_mix"
base_dir = "/home/taiga/ClassHyPer-master/"
log_dir = os.path.join(base_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
from_dir = "/home/taiga/ClassHyPer-master/discri_mix"
os.mkdir(log_dir)

with open(log_dir + "/loss.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc"])

data_transforms = {
    # 学習時の Transform
    "train": transforms.Compose(
        [
            #transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    # 推論時の Transform
    "val": transforms.Compose(
        [
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}

def data_transforms_test(img):

    data_transforrms_test = transforms.Compose(
        [
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    return data_transforrms_test(img)

img_datasets = {
    x: datasets.ImageFolder(dataset_dir + "/" +  x, data_transforms[x])
    for x in ["train", "val"]
}

dataloaders = {
    x: data.DataLoader(img_datasets[x], batch_size=16, shuffle=True, num_workers=1)
    for x in ["train", "val"]
}

def get_device(gpu_id=-1):
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    else:
        return torch.device("cpu")


device = get_device(gpu_id=0)

def train(model, criterion, optimizer, scheduler, dataloaders, device, n_epochs):
    """指定したエポック数だけ学習する。
    """
    with open(log_dir + "/loss.csv", 'a') as f:
        writer = csv.writer(f)
        history = []
        for epoch in range(n_epochs):
            info = train_on_epoch(
                model, criterion, optimizer, scheduler, dataloaders, device
            )
            info["epoch"] = epoch + 1
            history.append(info)

            print(
                f"epoch {info['epoch']:<2} "
                f"[train] loss: {info['train_loss']:.6f}, accuracy: {info['train_accuracy']:.0%} "
                f"[test] loss: {info['val_loss']:.6f}, accuracy: {info['val_accuracy']:.0%}"
            )
            writer.writerow((info['train_loss'], info['train_accuracy'], info['val_loss'], info['val_accuracy']))

        return history

def train_on_epoch(model, criterion, optimizer, scheduler, dataloaders, device):
    """1エポックだけ学習する学習する。
    """
    info = {}
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()  # モデルを学習モードに設定する。
        else:
            model.eval()  # モデルを推論モードに設定する。

        total_loss = 0
        total_correct = 0
        iter = 0
        for inputs, labels ,path in dataloaders[phase]:
            # データ及びラベルを計算を実行するデバイスに転送する。
            inputs, labels = inputs.to(device), labels.to(device)
            #print(path)
            #print(labels)
            #print(iter)
            # 学習時は勾配を計算するため、set_grad_enabled(True) で中間層の出力を記録するように設定する。
            with torch.set_grad_enabled(phase == "train"):
                # 順伝搬を行う。
                outputs = model(inputs)
                # 確率の最も高いクラスを予測ラベルとする。
                preds = outputs.argmax(dim=1)

                # 損失関数の値を計算する。
                loss = criterion(outputs, labels)

                if phase == "train":
                    # 逆伝搬を行う。
                    optimizer.zero_grad()
                    loss.backward()

                    # パラメータを更新する。
                    optimizer.step()

            # この反復の損失及び正答数を加算する。
            total_loss += float(loss)
            total_correct += int((preds == labels).sum())
            iter += 1

        if phase == "train":
            # 学習率を調整する。
            scheduler.step()

        # 損失関数の値の平均及び精度を計算する。
        info[f"{phase}_loss"] = total_loss / len(dataloaders[phase].dataset)
        info[f"{phase}_accuracy"] = total_correct / len(dataloaders[phase].dataset)

    return info

# ResNet-18 を作成する。
model_ft = models.resnet18(pretrained=True)

# 出力層の出力数を ImageNet の 1000 からこのデータセットのクラス数である 2 に置き換える。
model_ft.fc = nn.Linear(model_ft.fc.in_features, 2)

model_ft.load_state_dict(torch.load("/home/taiga/ClassHyPer-master/model_weight.pth"))

# モデルを計算するデバイスに転送する。
model_ft = model_ft.to(device)

# 損失関数を作成する。
criterion = nn.CrossEntropyLoss()


dataloaders = {
    x: data.DataLoader(img_datasets[x], batch_size=1, shuffle=True, num_workers=1)
    for x in ["train", "val"]
}

def show_prediction(model, transform, dataloaders):
    accuracy = 0
    precision = 0
    recall = 0
    precision_n = 0
    recall_n = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for inputs, labels ,path in dataloaders["val"]:
            # データ及びラベルを計算を実行するデバイスに転送する。
        inputs, labels = inputs.to(device), labels.to(device)
        # 1. PIL Image を標準化したテンソルにする。
        # 2. バッチ次元を追加する。 (C, H, W) -> (1, C, H, W)
        # 3. 計算するデバイスに転送する。

        with torch.no_grad():
            # 順伝搬を行う。
            outputs = model(inputs)

            # 確率の最も高いクラスを予測ラベルとする。
            class_id = int(outputs.argmax(dim=1)[0])
            print(class_id)

            if class_id == labels:
                f = open(log_dir + '/correct.txt', 'a')
                f.write(str(path) + ": " + str(class_id) + "\n")
                f.close
                if class_id == 0:
                    TN += 1
                else:
                    TP += 1

            else :
                f = open(log_dir + '/incorrect.txt', 'a')
                f.write(str(path) + ": " + str(class_id) + "\n")
                f.close
                if class_id == 0:
                    FP += 1
                else:
                    FN += 1

    accuracy = TP + TN / (TP + TN + FN + FP)
    precision = TP /(TP + FP)
    recall = TP /(TP + FN)
    precision_n = TN /(TN + FN)
    recall_n = TN /(TN + FP)
    print("accuracy:" + str(accuracy) + ", precision:" + str(precision) + ", recall:" + str(recall) + ", precision_negative:" + str(precision_n) + ", recall_nageative:" + str(recall_n))

show_prediction(model_ft, transforms, dataloaders)

