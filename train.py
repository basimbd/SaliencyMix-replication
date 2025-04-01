import os
import torch
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from models.resnet import ResNet50
from evaluate import evaluate
from const import cifar10_mean, cifar10_std
from utils import get_current_timestamp_string

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)])

train_dataset = datasets.CIFAR10(root='data/', train=True, transform=train_transform, download=True)
test_dataset = datasets.CIFAR10(root='data/', train=False, transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

model = ResNet50()
model = torch.nn.DataParallel(model).cuda()
loss_fn = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)    # milestone 60,12,160 according to paper

os.makedirs('logs', exist_ok=True)
log_filename = 'logs/' + 'cifar-10_resnet-50' + '.txt'

best_accuracy = 0
best_acc_epoch = 0

for epoch in range(1, 200+1):
    loss_accum = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    model.train()
    for i, (images, labels) in enumerate(progress_bar, start=1):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        model.zero_grad()
        pred = model(images)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()

        loss_accum += loss.item()

        _, pred_idxs = torch.max(pred.data, 1)
        total += labels.size(0)
        correct += (pred_idxs == labels.data).sum().item()
        accuracy = (correct / total)*100

        progress_bar.set_postfix(
            loss_avg='%.3f' % (loss_accum / i),
            acc='%.3f' % accuracy)

    test_acc = evaluate(model, test_loader=test_loader)
    tqdm.write('test_acc: %.3f' % (test_acc))

    scheduler.step()

    log_row = f"epoch: {str(epoch)},\ttrain_acc: {str(accuracy)},\ttest_acc: {str(test_acc)}"
    with open(log_filename, 'a') as log_f:
        log_f.write(f"{log_row}\n")

    if(test_acc>best_accuracy):
        best_accuracy = test_acc
        best_acc_epoch = epoch
        torch.save(model.state_dict(), 'checkpoints/' + f'cifar-10_resnet-50_best_{get_current_timestamp_string()}' + '.pt')

os.makedirs('checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/' + f'cifar-10_resnet-50_final_{get_current_timestamp_string()}' + '.pt')

f = open(f"best_accuracy_{get_current_timestamp_string()}.txt", "a+")
f.write('best acc: %.3f at epoch: %d \r\n' % (best_accuracy, best_acc_epoch))
f.close()