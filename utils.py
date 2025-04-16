from datetime import datetime
import torch
from torchvision import datasets, transforms
from models.resnet import ResNet50
from models.wideresnet import WideResNet
from cutout import Cutout
from const import cifar10_mean, cifar10_std, cifar100_mean, cifar100_std, imagenet_mean, imagenet_std, imagenet_eigvalue, imagenet_eigvector

def get_current_timestamp_string():
    return datetime.now().strftime('%Y%m%d')

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
    
def get_num_classes(dataset: str) -> int:
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'tiny-imagenet':
        return 200
    elif dataset == 'imagenet':
        return 1000
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def get_model(args):
    num_classes = get_num_classes(args.dataset)
    if args.model == 'resnet18':
        model = ResNet50(num_classes).cuda()
    elif args.model == 'resnet50':
        model = ResNet50(num_classes).cuda()
    elif args.model == 'resnet101':
        model = ResNet50(num_classes).cuda()
    elif args.model == 'wideresnet':
        model = WideResNet(depth=28, widen_factor=10, num_classes=num_classes, dropout_rate=0.3).cuda()
    else:
        raise ValueError(f"{args.model} model is not supported.")
    return model

def get_normalize_transforms(dataset: str):
    if dataset == 'cifar10':
        return cifar10_mean, cifar10_std
    elif dataset == 'cifar100':
        return cifar100_mean, cifar100_std
    elif 'imagenet' in dataset:
        return imagenet_mean, imagenet_std

def get_dataset(args, is_train: bool = True):
    mean, std = get_normalize_transforms(args.dataset)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    if args.trad_augment and is_train:
        print("Using traditional augmentation")
        transform.transforms.append(transforms.RandomCrop(32, padding=4))
        transform.transforms.append(transforms.RandomHorizontalFlip())
    if args.cutout:
        transform.transforms.append(Cutout(1, args.cutout_length))
    if args.dataset == 'cifar10':
        return datasets.CIFAR10(root='data/', train=is_train, transform=transform, download=True)
    elif args.dataset == 'cifar100':
        return datasets.CIFAR100(root='data/', train=is_train, transform=transform, download=True)
    elif 'imagenet' in args.dataset:
        transform = transforms.Compose([])
        if is_train:
            data_dir = 'imagenet/train' if args.dataset == 'imagenet' else 'tiny-imagenet/train'
            transform.transforms.append(transforms.RandomResizedCrop(224))
            transform.transforms.append(transforms.RandomHorizontalFlip())
            transform.transforms.append(transforms.ToTensor())
            transform.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
            transform.transforms.append(Lighting(0.1, imagenet_eigvalue, imagenet_eigvector))
        else:
            data_dir = 'imagenet/val' if args.dataset == 'imagenet' else 'tiny-imagenet/val'
            transform.transforms.append(transforms.Resize(256))
            transform.transforms.append(transforms.CenterCrop(224))
            transform.transforms.append(transforms.ToTensor())
        transform.transforms.append(transforms.Normalize(mean, std))
        return datasets.ImageFolder(data_dir, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

if __name__ == '__main__':
    print(get_current_timestamp_string())


