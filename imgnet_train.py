from const import imagenet_mean, imagenet_std, imagenet_eigvalue, imagenet_eigvactor
from torchvision import transforms
from torchvision import datasets
from utils import Lighting
from models.resnet import ResNet_imagenet as ResNet
import torch
import torch.nn as nn
import time
import numpy as np
import random

from saliency import get_salient_coordinates


best_error1, best_error5  = 100, 100


def main():
    global best_error1, best_error5
    
    train_directory = 'imagenet/train'
    val_directory = 'imagenet/val'


    normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

    jittering = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)

    lightening = Lighting(0.1, imagenet_eigvalue, imagenet_eigvactor)

    train_dataset = datasets.ImageFolder(
        train_directory,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            jittering,
            transforms.ToTensor(),
            lightening,
            normalize,
        ]))
    
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True, sampler=train_sampler) 
    
    val_dataset = datasets.ImageFolder(
        val_directory,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_directory, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, sampler=val_sampler)
    
    num_ephochs = 300


    net = ResNet(num_classes=200)
    net = torch.nn.DataParallel(net).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)

    for epoch in range(300):
        lr_udate = 0.1 * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_udate

        
        loss_train = train(net, train_loader, criterion, optimizer, epoch)

        error1, error5, val_loss = evaluate(net, val_loader, criterion, epoch)
        print(f'Validation Error@1: {error1:.2f}%, Error@5: {error5:.2f}%')


        if error1 < best_error1:
            best_error1 = error1
            best_error5 = error5
            best_epoch = epoch + 1
            print(f'Best Error@1: {best_error1:.2f}%, Error@5: {best_error5:.2f}% at epoch {best_epoch}')

            torch.save(net.state_dict(), f'./checkpoint/best_model_epoch_{best_epoch}.pth')

            print(f'Model saved at epoch {best_epoch}')

            log_file = open('log_imagenet_resnet50.txt', 'a+')
            log_file.write(f'Epoch: {epoch+1}, Error@1: {error1:.2f}%, Error@5: {error5:.2f}%\n')
            log_file.close()
    
    log_file = open('log_imagenet_resnet50.txt', 'a+')
    log_file.write(f'Best Error@1: {best_error1:.2f}%, Error@5: {best_error5:.2f}% at epoch {best_epoch}\n')
    log_file.write(f'Final Training Loss: {loss_train:.4f}\n')
    log_file.close()
        


def train(net, train_loader, criterion, optimizer, epoch):
    net.train()
    batch_time = AvgMeter('Time', ':6.3f')
    data_time = AvgMeter('Data', ':6.3f')
    losses = AvgMeter('Loss', ':.4e')
    top1 = AvgMeter('Acc@1', ':6.2f')
    top5 = AvgMeter('Acc@5', ':6.2f')

    end = time.time()

    lr_curr = get_current_lr(optimizer)

    for i, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda()
        labels = labels.cuda()

        random_int = np.random.rand(1)

        if random_int < 0.5:
            lamb = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(images.size()[0]).cuda()
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = get_salient_coordinates(images[rand_index[0]], lamb)

            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            lamb = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2])

            image_var = torch.autograd.Variable(images, requires_grad=True)
            target_a_var = torch.autograd.Variable(target_a, requires_grad=False)
            target_b_var = torch.autograd.Variable(target_b, requires_grad=False)

            outputs = net(image_var)
            loss = criterion(outputs, target_a_var) * lamb + criterion(outputs, target_b_var) * (1. - lamb)
        else:
            image_var = torch.autograd.Variable(images, requires_grad=True)
            target_var = torch.autograd.Variable(labels, requires_grad=False)

            outputs = net(image_var)
            loss = criterion(outputs, target_var)
        
        error1, error5 = accuracy(outputs.data, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(error1.item(), images.size(0))
        top5.update(error5.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0: 
            print(f'Epoch [{epoch+1}], Step [{i+1}/{len(train_loader)}], Loss: {losses.avg:.4f}, '
                  f'Acc@1: {top1.avg:.2f}%, Acc@5: {top5.avg:.2f}%, LR: {lr_curr:.6f}')
        
        return losses.avg

def evaluate(net, val_loader, criterion, epoch):
    net.eval()
    batch_time = AvgMeter('Time', ':6.3f')
    losses = AvgMeter('Loss', ':.4e')
    top1 = AvgMeter('Acc@1', ':6.2f')
    top5 = AvgMeter('Acc@5', ':6.2f')

    end = time.time()

    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            labels = labels.cuda()

            outputs = net(images)
            loss = criterion(outputs, labels)

            error1, error5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(error1.item(), images.size(0))
            top5.update(error5.item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

        print(f'Validation Loss: {losses.avg:.4f}, Acc@1: {top1.avg:.2f}%, Acc@5: {top5.avg:.2f}%')

        return top1.avg, top5.avg, losses.avg    

def get_current_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lrs += [param_group['lr']]
    return lrs[0]

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        result.append(wrong_k.mul_(100.0 / batch_size))
    return result


class AvgMeter(object):
   
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f'{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})'


if __name__ == '__main__':
    main()