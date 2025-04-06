from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from evaluate import evaluate
from const import cifar10_mean, cifar10_std
from utils import get_current_timestamp_string


import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime

from models.wideresnet import WideResNet
from models.wideresnet import conv_init

from torch.autograd import Variable
best_acc = 0

def main():
    # Hyper Parameter settings
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    # global best_acc
    start_epoch, num_epochs, batch_size, optim_type = 1, 200, 128, 'SGD'


    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]) # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    trainset = datasets.CIFAR10(root='data/', train=True, transform=transform_train, download=True)
    testset = datasets.CIFAR10(root='data/', train=False, transform=transform_test, download=True)

    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)


    def getNetwork():
    
        net = WideResNet(depth=28, widen_factor=10, num_classes=10, dropout_rate=0.3)
            
        file_name = 'wide-resnet-30 '
        

        return net, file_name

    
    print('\n[Phase 2] : Model setup')
   
    print('| Building net type Wideresnet...')
    net, file_name = getNetwork()
    net.apply(conv_init)

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

# Training
    def train(epoch):
        net.train()
        net.training = True
        train_loss = 0
        correct = 0
        total = 0
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, 0.1))
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)               # Forward Propagation
            loss = criterion(outputs, targets)  # Loss
            loss.backward()  # Backward Propagation
            optimizer.step() # Optimizer update

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                    %(epoch, num_epochs, batch_idx+1,
                        (len(trainset)//batch_size)+1, loss.item(), 100.*correct/total))
            sys.stdout.flush()

    def test(epoch):
        global best_acc
        net.eval()
        net.training = False
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

            # Save checkpoint when best model
            acc = 100.*correct/total
            print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))

            if acc > best_acc:
                print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
                state = {
                        'net':net.module if use_cuda else net,
                        'acc':acc,
                        'epoch':epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                save_point = './checkpoint/'+ "test_wide"
                if not os.path.isdir(save_point):
                    os.mkdir(save_point)
                torch.save(state, save_point+file_name+'.t7')
                best_acc = acc   

# Model
    


    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = 0.1' )
    print('| Optimizer = ' + str(optim_type))

    elapsed_time = 0
    for epoch in range(start_epoch, start_epoch+num_epochs):
        start_time = time.time()

        train(epoch)
        test(epoch)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d'  %(elapsed_time))

    print('\n[Phase 4] : Testing model')
    print('* Test results : Acc@1 = %.2f%%' %(best_acc))

if __name__ == '__main__':
    # multiprocessing.freeze_support()  # Only needed for frozen executables
    main()