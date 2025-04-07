# CIFAR-10 normalization values: https://github.com/kuangliu/pytorch-cifar/issues/19

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

imagenet_eigvalue = [0.2175, 0.0188, 0.0045]
imagenet_eigvactor = [
    [-0.5675, 0.7202, 0.4009],
    [0.5808, 0.0045, 0.8140],
    [0.5836, 0.6948, -0.4203]
]

