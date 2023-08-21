import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn

from models import *
from adamPre import AdamPre
import argparse
from attacks import pgd_linf, fgsm_attack
from dataloader import *
import torch.optim as optim
import torch.nn.functional as F

parser = argparse.ArgumentParser(
    description='Trains a PointCloud Classifier with adversarial attacks',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--model',
    '-m',
    type=str,
    default='PointNet',
    choices=['PointNet', 'DGCNN'],
    help='Choose architecture.')
parser.add_argument(
    '--epochs', '-e', type=int, default=50, help='Number of epochs to train.')
parser.add_argument(
    '--Tnet',
    '-tnet',
    action='store_false', #default value is true
    help='With or without Tnet add --tnet to change it to withouttnet.')
parser.add_argument(
    '--training',
    '-tr',
    type=str,
    default='',
    choices=['fgsm_attack', 'pgd_linf','mixed'],
    help='Choose training and also add attack.')
parser.add_argument(
    '--attack',
    '-at',
    type=str,
    default='',
    choices=['fgsm', 'pgd'],
    help='Choose attack and if needed add mixed training.')
parser.add_argument(
    '--Prediction',
    '-pred',
    action='store_true', #default value is false
    help='With or without Prediction add -pred to change to prediction.')
parser.add_argument(
    '--learning_rate',
    '-lr',
    type=float,
    default=0.01,
    help='Initial learning rate.')
parser.add_argument(
    '--step_size',
    '-ss',
    type=float,
    default=0.01,
    help='stepsize for attacks.')
parser.add_argument(
    '--steps',
    '-s',
    type=float,
    default=5,
    help='total steps for attacks.')
parser.add_argument(
    '--epsilon',
    '-eps',
    type=float,
    default=0.1,
    help='epsilon for attacks.')
parser.add_argument(
    '--num_workers',
    '-nworkers',
    type=int,
    default=0,
    help='Number of pre-fetching threads.')
parser.add_argument(
    '--batch-size', '-b', type=int, default=32, help='Batch size.')
parser.add_argument(
    '--delta',
    '-del',
    type=float,
    default=1,
    help='delta or multiplication factor for prediction step.')

args = parser.parse_args()


def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.001, x = args.Tnet): #loss function when using PointNet Model
    criterion = torch.nn.NLLLoss()
    if x == True:
        bs=outputs.size(0)
        d = m64x64.size()[1]
        I = torch.eye(d)[None, :, :]
        if m64x64.is_cuda:
            I = I.cuda()
            #I = I
        losse = torch.mean(torch.norm(torch.bmm(m64x64, m64x64.transpose(2, 1)) - I, dim=(1, 2)))
        loss = criterion(outputs, labels) + alpha * losse
    else:
        loss = criterion(outputs, labels)
    return loss

class Lit(pl.LightningModule): #class which helps in training using pytorch lightning
    def __init__(self,model = args.model,prediction=args.Prediction,learning_rate = args.learning_rate,eps=args.epsilon, attack= args.attack,num_steps = args.steps, step_size = args.step_size, Tnet = args.Tnet):
    # def __init__(self,*args):
        
        super().__init__()
        self.args = args
        if args.model == 'PointNet':
            self.model = PointNet()
        elif args.model == 'DGCNN':
            self.model = DGCNN()
        self.save_hyperparameters()
        self.learning_rate = args.learning_rate
        if args.model == 'PointNet':
            self.criterion = torch.nn.NLLLoss()
        elif args.model == 'DGCNN':
            self.criterion = torch.nn.NLLLoss()
        
        self.attack = args.attack
        self.eps = args.epsilon
        self.num_steps = args.steps
        self.step_size = args.step_size
        self.Tnet = args.Tnet
        if args.Prediction == True: 
            self.automatic_optimization=False #used when prediction step is used so that pytorch lightning optimization is set to false
            self.firstTime = True
       
    def forward(self, x, Tnet):
        if args.model == 'PointNet':
            return self.model(x, self.Tnet)
        elif args.model == 'DGCNN':
            return self.model(x)
    def configure_optimizers(self):
        if args.Prediction == True:
            opt = AdamPre(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999),d = args.delta)
        else :
            opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return opt
    
    def on_validation_model_eval(self, *args, **kwargs): #This is must to work with gradients in validation with pytorch lightning
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def training_step(self, batch, batch_idx): #Training Loop
        inputs, labels = batch['pointcloud'].float(), batch['category']
        opt1 = self.optimizers()
        if args.model == 'PointNet': #this is used to set which model is used in attacks.py
            m = True
        else:
            m = False
        if self.attack == 'pgd':
            adv_inputs = pgd_linf(self.model, inputs, labels,self.criterion,self.num_steps,self.step_size,self.eps,self.Tnet,m)
        elif self.attack == 'fgsm':
            adv_inputs = fgsm_attack(self.model,self.criterion,inputs,labels,self.eps,self.Tnet, m)
        else:
            adv_inputs = torch.zeros_like(inputs)

        if args.model == 'PointNet':
            if args.Prediction == True: #we just use prediction step for generating adversaraial examples
                if args.training == 'mixed':
                    if not self.firstTime:
                        opt1.stepLookAhead()
                    if self.attack == 'pgd':
                        adv_inputs = pgd_linf(self.model, inputs, labels,self.criterion,self.num_steps,self.step_size,self.eps,self.Tnet,m)
                    elif self.attack == 'fgsm':
                        adv_inputs = fgsm_attack(self.model,self.criterion,inputs,labels,self.eps,self.Tnet, m)
                    else:
                        adv_inputs = torch.zeros_like(inputs)
                    if not self.firstTime: #first time steplookhead doesnt contain values
                        opt1.restoreStepLookAhead() 
                    outputs1, cm3x3, cm64x64 = self.model((inputs).transpose(1,2), self.Tnet) #clean samples
                    outputs2, am3x3, am64x64 = self.model((inputs+adv_inputs).transpose(1,2), self.Tnet) #adversarial samples
                    opt1.zero_grad()
                    loss1 = pointnetloss(outputs1, labels, cm3x3, cm64x64, self.Tnet)
                    loss2 = pointnetloss(outputs2, labels, am3x3, am64x64, self.Tnet)
                    loss = loss1 + loss2
                    self.manual_backward(loss)
                    opt1.step()       
                    self.firstTime = False
                    _, preds1 = torch.max(outputs1.data, 1)
                    _, preds2 = torch.max(outputs2.data, 1)
                    acc1 = (preds1 == labels).float().mean() * 100
                    acc2 = (preds2 == labels).float().mean() * 100
                    self.log("train_acc", acc1, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    self.log("pgd_train_acc", acc2, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    self.log("clean_train_loss", loss1, on_epoch=True,sync_dist=True)
                    self.log("pgd_train_loss", loss2, on_epoch=True,sync_dist=True)
                    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    return loss
                else:
                    if not self.firstTime:
                        opt1.stepLookAhead()
                    if self.attack == 'pgd':
                        # print('just pgd training')
                        adv_inputs = pgd_linf(self.model, inputs, labels,self.criterion,self.num_steps,self.step_size,self.eps,self.Tnet,m)
                    elif self.attack == 'fgsm':
                        adv_inputs = fgsm_attack(self.model,self.criterion,inputs,labels,self.eps,self.Tnet, m)
                    else:
                        adv_inputs = torch.zeros_like(inputs)
                    if not self.firstTime:
                        opt1.restoreStepLookAhead()
                    # outputs1, cm3x3, cm64x64 = self.model((inputs).transpose(1,2), self.Tnet)
                    outputs2, am3x3, am64x64 = self.model((inputs+adv_inputs).transpose(1,2), self.Tnet)
                    opt1.zero_grad()
                    loss2 = pointnetloss(outputs2, labels, am3x3, am64x64, self.Tnet)
                    self.manual_backward(loss2)
                    opt1.step()        
                    self.firstTime = False
                    self.log("train_loss", loss2, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    return loss2      
            else:
                if args.training == 'mixed':
                    outputs1, cm3x3, cm64x64 = self.model((inputs).transpose(1,2), self.Tnet) #clean samples
                    outputs2, am3x3, am64x64 = self.model((inputs+adv_inputs).transpose(1,2), self.Tnet) #adversarial samples
                    loss1 = pointnetloss(outputs1, labels, cm3x3, cm64x64, self.Tnet)
                    loss2 = pointnetloss(outputs2, labels, am3x3, am64x64, self.Tnet)
                    loss = loss1 + loss2
                    _, preds1 = torch.max(outputs1.data, 1)
                    _, preds2 = torch.max(outputs2.data, 1)
                    acc1 = (preds1 == labels).float().mean() * 100
                    acc2 = (preds2 == labels).float().mean() * 100
                    self.log("train_acc", acc1, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    self.log("pgd_train_acc", acc2, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    self.log("clean_train_loss", loss1, on_epoch=True,sync_dist=True)
                    self.log("pgd_train_loss", loss2, on_epoch=True,sync_dist=True)
                    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    return loss
                else:
                    outputs, m3x3, m64x64 = self.model((inputs+adv_inputs).transpose(1,2), self.Tnet)
                    loss = pointnetloss(outputs, labels, m3x3, m64x64, self.Tnet)
                    _, preds = torch.max(outputs.data, 1)
                    acc = (preds == labels).float().mean() * 100
                    self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    return loss
        if args.model == 'DGCNN':
            if args.Prediction == True:
                if args.training == 'mixed':
                    if not self.firstTime:
                        opt1.stepLookAhead()
                    if self.attack == 'pgd':
                        adv_inputs = pgd_linf(self.model, inputs, labels,self.criterion,self.num_steps,self.step_size,self.eps,self.Tnet,m)
                    elif self.attack == 'fgsm':
                        adv_inputs = fgsm_attack(self.model,self.criterion,inputs,labels,self.eps,self.Tnet, m)
                    else:
                        adv_inputs = torch.zeros_like(inputs)
                    if not self.firstTime: #first time steplookhead doesnt contain values
                        opt1.restoreStepLookAhead() 
                    outputs1 = self.model((inputs).permute(0, 2, 1))
                    outputs2 = self.model((inputs+adv_inputs).permute(0, 2, 1))
                    opt1.zero_grad()
                    loss1 = self.criterion(outputs1,labels)
                    loss2 = self.criterion(outputs2,labels)
                    loss = loss1 + loss2
                    self.manual_backward(loss)
                    opt1.step()        
                    self.firstTime = False
                    _, preds1 = torch.max(outputs1.data, 1)
                    _, preds2 = torch.max(outputs2.data, 1)
                    acc1 = (preds1 == labels).float().mean() * 100
                    acc2 = (preds2 == labels).float().mean() * 100
                    self.log("clean_train_acc", acc1, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    self.log("pgd_train_acc", acc2, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    self.log("clean_train_loss", loss1, on_epoch=True,sync_dist=True)
                    self.log("pgd_train_loss", loss2, on_epoch=True,sync_dist=True)
                    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    return loss
                else:
                    if not self.firstTime:
                        opt1.stepLookAhead()
                    if self.attack == 'pgd':
                        adv_inputs = pgd_linf(self.model, inputs, labels,self.criterion,self.num_steps,self.step_size,self.eps,self.Tnet,m)
                    elif self.attack == 'fgsm':
                        adv_inputs = fgsm_attack(self.model,self.criterion,inputs,labels,self.eps,self.Tnet, m)
                    else:
                        adv_inputs = torch.zeros_like(inputs)
                    if not self.firstTime: #first time steplookhead doesnt contain values
                        opt1.restoreStepLookAhead() 
                    outputs2 = self.model((inputs+adv_inputs).permute(0, 2, 1))
                    opt1.zero_grad()
                    loss2 = self.criterion(outputs2, labels)
                    self.manual_backward(loss2)
                    opt1.step()        
                    self.firstTime = False
                    self.log("train_loss", loss2, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    return loss2       
            else:
                if args.training == 'mixed':
                    outputs1 = self.model((inputs).permute(0, 2, 1))
                    outputs2 = self.model((inputs+adv_inputs).permute(0, 2, 1))
                    loss1 = self.criterion(outputs1,labels)
                    loss2 = self.criterion(outputs2,labels)
                    loss = loss1 + loss2
                    _, preds1 = torch.max(outputs1.data, 1)
                    _, preds2 = torch.max(outputs2.data, 1)
                    acc1 = (preds1 == labels).float().mean() * 100
                    acc2 = (preds2 == labels).float().mean() * 100
                    self.log("clean_train_acc", acc1, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    self.log("pgd_train_acc", acc2, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    self.log("clean_train_loss", loss1, on_epoch=True,sync_dist=True)
                    self.log("pgd_train_loss", loss2, on_epoch=True,sync_dist=True)
                    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    return loss
                else:
                    outputs = self.model((inputs+adv_inputs).permute(0, 2, 1))
                    loss = self.criterion(outputs, labels)
                    _, preds = torch.max(outputs.data, 1)
                    acc = (preds == labels).float().mean() * 100
                    self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    return loss
        
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['pointcloud'].float(), batch['category']
        if args.model == 'PointNet':
            m = True
        else:
            m = False
        if self.attack == "pgd" :
            adv_inputs = pgd_linf(self.model, inputs, labels,self.criterion,self.num_steps,self.step_size,self.eps,self.Tnet,m)
        elif self.attack == "fgsm" :
            adv_inputs = fgsm_attack(self.model,self.criterion,inputs,labels,self.eps,self.Tnet, m)
        else:
            adv_inputs = torch.zeros_like(inputs)

        if args.model == 'PointNet':
                if args.training == 'mixed':
                    outputs1, cm3x3, cm64x64 = self.model((inputs).transpose(1,2), self.Tnet)
                    outputs2, am3x3, am64x64 = self.model((inputs+adv_inputs).transpose(1,2), self.Tnet)
                    loss1 = self.criterion(outputs1, labels)
                    loss2 = self.criterion(outputs2, labels)
                    loss = loss1 + loss2
                    _, preds1 = torch.max(outputs1.data, 1)
                    _, preds2 = torch.max(outputs2.data, 1)
                    acc1 = (preds1 == labels).float().mean() * 100
                    acc2 = (preds2 == labels).float().mean() * 100
                    self.log("validation_acc", acc1, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    self.log("pgd_validation_acc", acc2, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    self.log("clean_validation_loss", loss1, on_epoch=True,sync_dist=True)
                    self.log("pgd_validation_loss", loss2, on_epoch=True,sync_dist=True)
                    self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    return loss
                else:
                    outputs, m3x3, m64x64 = self.model((inputs+adv_inputs).transpose(1,2), self.Tnet)
                    loss = self.criterion(outputs, labels)
                    _, preds = torch.max(outputs.data, 1)
                    acc = (preds == labels).float().mean() * 100
                    self.log("validation_acc", acc, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    return loss
        if args.model == 'DGCNN':
                if args.training == 'mixed':
                    outputs1 = self.model((inputs).permute(0, 2, 1))
                    outputs2 = self.model((inputs+adv_inputs).permute(0, 2, 1))
                    loss1 = self.criterion(outputs1,labels)
                    loss2 = self.criterion(outputs2,labels)
                    loss = loss1 + loss2
                    _, preds1 = torch.max(outputs1.data, 1)
                    _, preds2 = torch.max(outputs2.data, 1)
                    acc1 = (preds1 == labels).float().mean() * 100
                    acc2 = (preds2 == labels).float().mean() * 100
                    self.log("validation_train_acc", acc1, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    self.log("pgd_validation_acc", acc2, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    self.log("clean_validation_loss", loss1, on_epoch=True,sync_dist=True)
                    self.log("pgd_validation_loss", loss2, on_epoch=True,sync_dist=True)
                    self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    return loss
                else:
                    outputs = self.model((inputs+adv_inputs).permute(0, 2, 1))
                    loss = self.criterion(outputs, labels)
                    _, preds = torch.max(outputs.data, 1)
                    acc = (preds == labels).float().mean() * 100
                    self.log("validation_acc", acc, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
                    return loss
if __name__ == '__main__':
    # model = Lit(args.learning_rate,args.epsilon,args.training,args.steps,args.step_size,args.Tnet)
    model = Lit(args.model,args.Prediction,args.learning_rate,args.epsilon,args.training,args.steps,args.step_size,args.Tnet)
    train_ds = PointCloudData(path, transform=train_transforms) #Train dataset
    # train_ds = PointCloudData(path)
    test_ds = PointCloudData(path,test= True,folder='test',transform=train_transforms) #test dataset
    valid_ds = PointCloudData(path, valid=True, folder='val', transform=train_transforms)#validation dataset
    
    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers) #Train loader
    valid_loader = DataLoader(dataset=valid_ds, batch_size=args.batch_size,num_workers=args.num_workers) #Valid Loader
    test_loader = DataLoader(dataset=test_ds,batch_size=args.batch_size) #Test Loader

    #model = Lit(args)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="validation_loss",
        mode="min",
        filename="checkpoint-{epoch:02d}-{validation_loss:.2f}-{validation_acc:.2f}-{train_acc:.2f}",) #change name if needed
    trainer = pl.Trainer(accelerator="gpu", devices=[0],max_epochs = args.epochs, callbacks=[checkpoint_callback],strategy="ddp_spawn")
    trainer.fit(model,train_loader,valid_loader)