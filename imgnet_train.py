from const import imagenet_mean, imagenet_std, imagenet_eigvalue, imagenet_eigvector, cifar100_mean, cifar100_std
from torchvision import transforms
from torchvision import datasets
from utils import Lighting, get_current_timestamp_string
from tqdm import tqdm
from models.resnet import ResNet_imagenet as ResNet_imagenet
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import time
import numpy as np
import random
from evaluate import evaluate, evaluate_w_top_k
from saliency import get_salient_coordinates


best_error1, best_error5  = 100, 100


def main():
    global best_error1, best_error5
    
    train_directory = 'imagenet/train'
    val_directory = 'imagenet/val'


    normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

    jittering = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)

    lightening = Lighting(0.1, imagenet_eigvalue, imagenet_eigvector)

    train_dataset = datasets.ImageFolder(
        train_directory,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),            
            transforms.ToTensor(),
            jittering,
            lightening,
            normalize,
        ]))
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True, sampler=None) 
    print("Train dataset size: ", len(train_loader.dataset))
    
    val_dataset = datasets.ImageFolder(
        val_directory,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    

    net = ResNet_imagenet(numberofclass=1000)
    

    net = nn.DataParallel(net).cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, 
    weight_decay=1e-4, nesterov=True)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=[75, 150, 225], gamma=0.1)

    for epoch in range(0, 300):

        progress_bar = tqdm(train_loader)
        loss_train = train(net, progress_bar, criterion, optimizer, epoch)

        error1, error5, val_loss = evaluate_w_top_k(model=net, test_loader=val_loader, loss_fn=criterion)

        tqdm.write(f'Epoch [{epoch+1}/{300}], Val Loss: {val_loss:.4f}, Err@1: {error1:.2f}%, Err@5: {error5:.2f}%')

        scheduler.step()

        best_error1 = min(best_error1, error1)

        if error1 <= best_error1:
            best_error5 = error5
            torch.save(net.state_dict(), f'checkpoints/imagenet/resnet50/base_model_best_{get_current_timestamp_string()}.pth')


def train(net, progress_bar, criterion, optimizer, epoch):
    
    losses = AvgMeter()
    top1 = AvgMeter()
    top5 = AvgMeter()

    loss_accum = 0.
    net.train()


    for i, (images, labels) in enumerate(progress_bar, start=1):
        progress_bar.set_description(f"Epoch [{epoch+1}/{300}]")

        images = images.cuda()
        labels = labels.cuda()

        if torch.rand(1).item() < 0.5:
            lamb = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(images.size(0)).cuda()
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bbx2, bby1, bby2 = get_salient_coordinates(images[rand_index[0]], lamb)

            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            lamb = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1)* images.size(-2))

            images.requires_grad_(True)

            outputs = net(images)
            loss = criterion(outputs, target_a) * lamb + criterion(outputs, target_b) * (1. - lamb)
        else:
            images.requires_grad_(True)
            outputs = net(images)
            loss = criterion(outputs, labels)
        
        error1, error5 = accuracy(outputs.data, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(error1.item(), images.size(0))
        top5.update(error5.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_accum += loss.item()

        progress_bar.set_postfix(
            train_loss='%.3f' % (loss_accum / i),
            err_1='%.3f' % error1,
            err_5='%.3f' % error5)
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


    # print(f'Epoch: {epoch}, Validation Loss: {losses.avg:.4f}, Err@1: {top1.avg:.2f}%, Err@5: {top5.avg:.2f}%')

    return top1.avg, top5.avg, losses.avg    

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