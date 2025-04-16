import argparse
from const import imagenet_mean, imagenet_std, imagenet_eigvalue, imagenet_eigvector
from torchvision import transforms
from utils import get_dataset, get_model, get_current_timestamp_string
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from evaluate import evaluate, evaluate_w_top_k
from saliency import get_salient_coordinates


def main(args):
    best_error1, best_error5  = 100, 100

    train_dataset = get_dataset(args, is_train=True)
    val_dataset = get_dataset(args, is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True) 
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    

    model = get_model(args)
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, 
    weight_decay=1e-4, nesterov=True)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=[75, 150, 225], gamma=0.1)


    for epoch in range(0, args.epochs):

        progress_bar = tqdm(train_loader)

        loss_train = train(model, progress_bar, criterion, optimizer, epoch, args.saliency_probab)

        error1, error5, val_loss = evaluate_w_top_k(model=model, test_loader=val_loader, loss_fn=criterion)

        tqdm.write(f'Epoch [{epoch+1}/{300}], Val Loss: {val_loss:.4f}, Err@1: {error1:.2f}%, Err@5: {error5:.2f}%')

        scheduler.step()

        best_error1 = min(best_error1, error1)

        if error1 <= best_error1:
            best_error5 = error5
            torch.save(model.state_dict(), f'checkpoints/imagenet/resnet50/base_model_best_{get_current_timestamp_string()}.pth')


def train(model, progress_bar, criterion, optimizer, epoch, mixing_probab=0.0):

    loss_accum = 0.
    model.train()


    for i, (images, labels) in enumerate(progress_bar, start=1):
        progress_bar.set_description(f"Epoch [{epoch+1}/{300}]")

        images = images.cuda()
        labels = labels.cuda()

        if torch.rand(1).item() < mixing_probab:
            rand_index = torch.randperm(images.size(0)).cuda()
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bbx2, bby1, bby2 = get_salient_coordinates(images[rand_index[0]], torch.distributions.Beta(1.0, 1.0).sample().item())

            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            lamb = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1)* images.size(-2))

            images.requires_grad_(True)

            outputs = model(images)
            loss = criterion(outputs, target_a) * lamb + criterion(outputs, target_b) * (1. - lamb)
        else:
            images.requires_grad_(True)
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        error1, error5 = accuracy(outputs.data, labels, topk=(1, 5))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_accum += loss.item()

        progress_bar.set_postfix(
            train_loss='%.3f' % (loss_accum / i),
            err_1='%.3f' % error1,
            err_5='%.3f' % error5)
    return loss_accum / i


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


if __name__ == '__main__':
    # # # # # # # # # # #
    # Arguments Parsing #
    # # # # # # # # # # #
    parser = argparse.ArgumentParser(description='SaliencyMix Replication Training Script')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50'], help='Model to use')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'tiny-imagenet'], help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--saliency_probab', type=float, default=0.0, help='Probability of applying SaliencyMix')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint to load')
    args = parser.parse_args()

    main(args)
