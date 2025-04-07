import pdb
from tqdm import tqdm
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np

from models.wideresnet import WideResNet 
# from models.wideresnet import conv_init


from const import cifar10_mean, cifar10_std, cifar100_mean, cifar100_std
# from evaluate import evaluate

from saliency import get_salient_coordinates

# import cv2


best_acc = 0
best_epoch = 1

def main():
    # Hyper Parameter settings
    use_cuda = torch.cuda.is_available()
    global best_acc
    global best_epoch
    print(use_cuda)
    start_epoch, num_epochs = 1, 200


    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
        # transforms.Normalize(cifar100_mean, cifar100_std),
    ]) # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
        # transforms.Normalize(cifar100_mean, cifar100_std),
    ])

    trainset = datasets.CIFAR10(root='data/', train=True, transform=transform_train, download=True)
    testset = datasets.CIFAR10(root='data/', train=False, transform=transform_test, download=True)

    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)


    net = WideResNet(depth=28, widen_factor=10, num_classes=10, dropout_rate=0.3) 

    net = torch.nn.DataParallel(net).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    net_optim = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(net_optim, milestones=[60, 120, 160], gamma=0.2)    # milestone 60,12,160 according to paper
    
    csv_file = 'logs/' + 'cifar-10_wide-resnet-101' + '.csv'

    def test(loader):
        global best_acc
        global best_epoch
        net.eval()
        net.training = False
        test_loss = 0
        correct = 0
        total = 0
        
        for images, labels in loader:
            images = images.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                pred = net(images)
                loss = criterion(pred, labels)

            test_loss += loss.item()
            
            prediction = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
        
        validation_accuracy = 100. * correct / total
        net.train()

        return validation_accuracy
    
# Model
        
    for epoch in range(start_epoch, start_epoch+num_epochs):
        start_time = time.time()

        xentropy_loss, loss_avg = 0.0, 0.0
        correct = 0.0
        total = 0.0
        progress_bar = tqdm(trainloader)

        for i, (images, labels) in enumerate(progress_bar, start=1):
            progress_bar.set_description('Epoch ' + str(epoch))

            images = images.cuda()
            labels = labels.cuda()

            r = np.random.rand(1)

            if r < 0.5:
                lamb = np.random.beta(1.0, 1.0)
                random_index = torch.randperm(images.size()[0]).cuda()
                labels_a, labels_b = labels, labels[random_index]
                x1, y1, x2, y2 = get_salient_coordinates( images[random_index[0]], lamb)
                images[:, :, x1:x2, y1:y2] = images[random_index, :, x1:x2, y1:y2]
                lamb = 1 - (x2 - x1) * (y2 - y1) / (images.size()[-1] * images.size()[-3])

                net.zero_grad()
                pred = net(images)

                xentropy_loss = criterion(pred, labels_a) * lamb + criterion(pred, labels_b) * (1. - lamb)
            
            else:
                net.zero_grad()
                pred = net(images)
                xentropy_loss = criterion(pred, labels)
            
            xentropy_loss.backward()
            net_optim.step()

            loss_avg += xentropy_loss.item()

            _, pred = torch.max(pred.data, 1)

            total += labels.size(0)

            correct += (pred == labels.data).sum().item()
            accuracy = 100 * correct / total

            progress_bar.set_postfix(
                loss_avg='%.3f' % (loss_avg / i),
                acc='%.3f' % accuracy)
            
        test_acc = test(testloader)
        tqdm.write('test_acc: %.3f' % (test_acc))

        scheduler.step(epoch=epoch)

        log_row = f"epoch: {str(epoch)},\t train_acc: {str(accuracy)},\t test_acc: {str(test_acc)}"
        with open(csv_file, 'a') as log_f:
            log_f.write(f"{log_row}\n")

        if(test_acc > best_acc):
            best_acc = test_acc
            best_epoch = epoch
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            torch.save(net.state_dict(), 'checkpoints/' + f'cifar-10_wide-resnet-101_best.pt')
    tqdm.write(f"Best accuracy: {best_acc:.2f} at epoch {best_epoch}")
           
          





if __name__ == '__main__':
    # multiprocessing.freeze_support()  # Only needed for frozen executables
    main()