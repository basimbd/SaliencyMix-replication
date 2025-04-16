import os
import torch
import time
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from models.resnet import ResNet50
from evaluate import evaluate
from utils import get_current_timestamp_string, get_dataset, get_model
from saliency import get_salient_coordinates
from cutmix import get_cutmix_coordinates

# # # # # # # # # # #
# Arguments Parsing #
# # # # # # # # # # #
parser = argparse.ArgumentParser(description='SaliencyMix Replication Training Script')
parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'wideresnet'], help='Model to use')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet'], help='Dataset to use')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--cutmix_probab', type=float, default=0.0, help='Probability of applying CutMix')
parser.add_argument('--saliency_probab', type=float, default=0.0, help='Probability of applying SaliencyMix')
parser.add_argument('--cutout', action='store_true', help='Use Cutout augmentation')
parser.add_argument('--cutout_length', type=int, default=16,  help='Cutout dimension')
parser.add_argument('--trad_augment', action='store_true', help='Use traditional augmentation')
parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint to load')
args = parser.parse_args()

# # # # # # # # # # # #
# Dataset Preparation #
# # # # # # # # # # # #
train_dataset = get_dataset(args, is_train=True)
test_dataset = get_dataset(args, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# # # # # # # # # # # # #
# Constants Preparation #
# # # # # # # # # # # # #
beta = 1.0
best_accuracy = 0
best_acc_epoch = 0

mixing_probab = max(args.saliency_probab, args.cutmix_probab)
assert (mixing_probab == 0 or (mixing_probab and args.saliency_probab == 0) or (mixing_probab and args.cutmix_probab == 0)), "Set probability for only one type of mixing"

mixing_type = "base_model"
if args.saliency_probab > 0:
    mixing_type = "saliMix"
elif args.cutmix_probab > 0:
    mixing_type = "cutmix"
elif args.cutout:
    mixing_type = "cutout"
if args.trad_augment:
    mixing_type += "_trad_augment"

log_filename = f'logs/{args.dataset}/{args.model}/{mixing_type}_{get_current_timestamp_string()}.txt'
os.makedirs(os.path.dirname(log_filename), exist_ok=True)

best_model_path = f'checkpoints/{args.dataset}/{args.model}/{mixing_type}_best_{get_current_timestamp_string()}.pt'
final_model_path = f'checkpoints/{args.dataset}/{args.model}/{mixing_type}_final_{get_current_timestamp_string()}.pt'
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

# # # # # # # # #
# Model Loading #
# # # # # # # # #
model = get_model(args)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)    # milestone 60,12,160 according to paper
start_epoch = 1
if args.checkpoint:
    print(f"Loading checkpoint from {args.checkpoint}")
    saved_checkpoint = torch.load(args.checkpoint, weights_only=True)
    start_epoch = saved_checkpoint['epoch'] + 1
    model.load_state_dict(saved_checkpoint['model_state_dict'])
    optimizer.load_state_dict(saved_checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(saved_checkpoint['scheduler_state_dict'])
    best_accuracy = saved_checkpoint['best_acc']
loss_fn = torch.nn.CrossEntropyLoss().cuda()

# # # # # # # # #
# Training Loop #
# # # # # # # # #
start = time.time()
for epoch in range(start_epoch, args.epochs+1):
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

        if beta > 0 and torch.rand(1).item() < mixing_probab:
            shuffle_index = torch.randperm(images.size(0)).cuda()
            original_labels = labels
            shuffled_labels = labels[shuffle_index]
            if args.saliency_probab:
                x1, x2, y1, y2 = get_salient_coordinates(images[shuffle_index[0]], torch.distributions.Beta(beta, beta).sample().item())
            elif args.cutmix_probab:
                x1, x2, y1, y2 = get_cutmix_coordinates(images[shuffle_index[0]], torch.distributions.Beta(beta, beta).sample().item())
            images[:, :, x1:x2, y1:y2] = images[shuffle_index, :, x1:x2, y1:y2]     # put selected region to the target image

            lamda = 1 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
            
            pred = model(images)
            # final_labels = λ * label_target + (1 − λ) * label_source
            loss = loss_fn(pred, original_labels) * lamda + loss_fn(pred, shuffled_labels) * (1. - lamda)
        else:
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
        model_save_dict = dict(
            epoch=epoch,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            best_acc=best_accuracy
        )
        torch.save(model_save_dict, best_model_path)

print(f"Total time: {(time.time() - start)/60:.2f} mins")

os.makedirs('checkpoints', exist_ok=True)
model_save_dict = dict(
    epoch=epoch,
    model_state_dict=model.state_dict(),
    optimizer_state_dict=optimizer.state_dict(),
    scheduler_state_dict=scheduler.state_dict(),
    best_acc=best_accuracy
)
torch.save(model_save_dict, final_model_path)

f = open(f"best_accuracy_{args.dataset}_{args.model}_{mixing_type}_{get_current_timestamp_string()}.txt", "a+")
f.write('best acc: %.3f at epoch: %d \r\n' % (best_accuracy, best_acc_epoch))
f.close()
