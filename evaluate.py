import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.resnet import ResNet50
from models.wideresnet import WideResNet
from const import cifar10_mean, cifar10_std

def evaluate(model=None, model_path=None, test_loader=None):
    assert model is not None or model_path is not None, "Either 'model' or 'model_path' should be provided"
    if test_loader is None:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std)])

        test_dataset = datasets.CIFAR10(root='data/', train=False, download=True, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    if model == "resnet":
        model = ResNet50()
        model.load_state_dict(torch.load(model_path))
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = WideResNet(depth=28, widen_factor=10, num_classes=10, input_channels=3, dor=0.3)
        model.load_state_dict(torch.load(model_path))
        model = torch.nn.DataParallel(model).cuda()   


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

if __name__ == '__main__':
    acc = evaluate(model="resnet")
    print(f"Accuracy: {acc:.2f}%")