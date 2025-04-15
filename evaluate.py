import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.resnet import ResNet50
from models.wideresnet import WideResNet
from const import cifar10_mean, cifar10_std, cifar100_mean, cifar100_std
from utils import get_dataset, get_model

def evaluate(model=None, model_path=None, test_loader=None, args=None):
    assert model is not None or model_path is not None, "Either 'model' or 'model_path' should be provided"
    if test_loader is None:
        test_dataset = get_dataset(args, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    if model is None:
        model = get_model(args)
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            pred = model(images)
            _, pred_idxs = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (pred_idxs == labels.data).sum().item()
    return 100 * correct / total

def evaluate_w_top_k(model=None, model_path=None, test_loader=None, k=5, args=None, loss_fn=None):
    assert model is not None or model_path is not None, "Either 'model' or 'model_path' should be provided"
    if test_loader is None:
        assert args is not None, "If 'test_loader' is None, 'args (--dataset)' must be provided"
        test_dataset = get_dataset(args, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    if model is None:
        model = get_model(args)
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
    loss_accum = 0.
    correct = 0
    total = 0
    correct_k = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            pred = model(images)
            loss = loss_fn(pred, labels)
            loss_accum += loss.item()
            _, pred_idxs = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (pred_idxs == labels.data).sum().item()
            _, pred_idxs = torch.topk(pred.data, k, 1)
            pred_idxs = pred_idxs.t()
            correct_k += pred_idxs.eq(labels.view(1, -1).expand_as(pred_idxs)).sum().item()
    return 100 * correct / total, 100 * correct_k / total, loss_accum / len(test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model evaluation script')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'wideresnet'], help='Model to use')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model file')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--cutout', action='store_true', help='Use Cutout augmentation')
    parser.add_argument('--trad_augment', action='store_true', help='Use traditional augmentation')

    args = parser.parse_args()
    if args.model_path is None:
        raise ValueError("--model_path must be provided")
    acc_1, acc_k = evaluate_w_top_k(model_path=args.model_path, args=args)
    print(f"Top-1 Accuracy: {acc_1:.2f}%\t Top-5 Accuracy: {acc_k:.2f}%")
