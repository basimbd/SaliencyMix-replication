from const import imagenet_mean, imagenet_std, imagenet_eigvalue, imagenet_eigvactor
from torchvision import transforms
from torchvision import datasets
from utils import Lighting
# from utils import ColorJitter
from models.resnet import ResNet_imagenet as ResNet_imagenet
import torch
import torch.nn as nn
import time
import numpy as np
import random

from evaluate import evaluate

from saliency import get_salient_coordinates


best_error1, best_error5  = 100, 100


def main():
    global best_error1, best_error5
    
    train_directory = 'imagenet/train'
    val_directory = 'imagenet/test'


    normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

    # jittering = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)

    lightening = Lighting(0.1, imagenet_eigvalue, imagenet_eigvactor)

    train_dataset = datasets.ImageFolder(
        train_directory,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),            
            transforms.ToTensor(),
            lightening,
            normalize,
        ]))
    
    # train_dataset = torch.utils.data.Subset(train_dataset, range(10))

    # train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, sampler=None) 
    
    val_dataset = datasets.ImageFolder(
        val_directory,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    

    net = ResNet_imagenet(numberofclass=200)   
    

    net = net.cuda()
    # net = torch.nn.DataParallel(net).cuda()

    print("Number of parameters: ", sum(p.numel() for p in net.parameters()))



    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, 
    weight_decay=1e-4, nesterov=True)


    for epoch in range(0, 300):
        
        lr = 0.1 * (0.1 ** (epoch // 75))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        
        loss_train = train(net, train_loader, criterion, optimizer, epoch)

        acc = evaluate(model=net, test_loader= val_loader)
        # print(f'Validation Error@1: {error1:.2f}%, Error@5: {error5:.2f}% , Loss: {val_loss:.4f}')
        print(f'Validation Acc: {100-acc:.2f}%')

        error1 = 100 - acc
        best_error1 = min(best_error1, 100.0-acc)
        # best_error1 = min(best_error1, error1)

        if error1 <= best_error1:
            # best_error5 = error5
            torch.save(net.state_dict(), f'checkpoint/resnet_imgnet_epoch_{epoch}.pth')
            
        # print(f' Current Best Error@1: {best_error1:.2f}%, Error@5: {best_error5:.2f}%')

        f = open('imagenet_results.txt', 'a+')
        # f.write(f'Epoch [{epoch+1}], Loss: {loss_train:.4f}, Best Error@1: {best_error1:.2f}%, Best Error@5: {best_error5:.2f}%\n')
        f.close()  

    # print(f'Best Error@1: {best_error1:.2f}%, Error@5: {best_error5:.2f}%')

    f = open('imagenet_results.txt', 'a+')
    # f.write(f'Best Error@1: {best_error1:.2f}%, Error@5: {best_error5:.2f}%\n')
    f.close()         
        


def train(net, train_loader, criterion, optimizer, epoch):
    
    losses = AvgMeter()
    top1 = AvgMeter()
    top5 = AvgMeter()

    net.train()


    lr_curr = get_current_lr(optimizer)[0]

    for i, (images, labels) in enumerate(train_loader):
        
        # print("tr img- ",images[0])
        # print("tr lab- ", labels)

        images = images.cuda()
        labels = labels.cuda()

        random_int = np.random.rand(1)

        if random_int < 0.5:
            lamb = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(images.size(0)).cuda()
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bbx2, bby1, bby2 = get_salient_coordinates(images[rand_index[0]], lamb)

            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            lamb = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1)* images.size(-2))

            image_var = torch.autograd.Variable(images, requires_grad=True)

            target_a_var = torch.autograd.Variable(target_a)
            target_b_var = torch.autograd.Variable(target_b)

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


        # if i % 10 == 0: 
        # print(f'Train - Epoch [{epoch+1}], Step [{i+1}/{len(train_loader)}], Loss: {losses.item():.4f}, '
                #   f'Acc@1: {top1.avg:.2f}%, Acc@5: {top5.avg:.2f}%, LR: {lr_curr:.6f}')
        
        return losses.avg

def validate(net, val_loader, criterion, epoch):
    
   
    losses = AvgMeter()
    top1 = AvgMeter()
    top5 = AvgMeter()

    net.eval()

    # with torch.no_grad():
    for i, (images, labels) in enumerate(val_loader):
        # print("vl img- ", images[0])
        # print("vl lab- ", labels)
        images = images.cuda()
        
        labels = labels.cuda()

        outputs = net(images)
        loss = criterion(outputs, labels)

        error1, error5 = accuracy(outputs.data, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(error1.item(), images.size(0))
        top5.update(error5.item(), images.size(0))


    print(f'Epoch: {epoch}, Validation Loss: {losses.avg:.4f}, Acc@1: {top1.avg:.2f}%, Acc@5: {top5.avg:.2f}%')

    return top1.avg, top5.avg, losses.avg    

def get_current_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

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
   
    def __init__(self):
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


if __name__ == '__main__':
    main()