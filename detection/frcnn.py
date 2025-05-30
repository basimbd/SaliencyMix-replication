import copy
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
import os
from detection.engine import train_one_epoch, evaluate  # from https://github.com/pytorch/vision/blob/main/references/detection/engine.py
import detection.utils as utils  # from https://github.com/pytorch/vision/blob/main/references/detection/utils.py
import detection.transforms as T  # from https://github.com/pytorch/vision/blob/main/references/detection/transforms.py
from detection.voc import get_voc_datasets
from models.resnet import ResNet_imagenet
from utils import get_current_timestamp_string


def freeze(layer):
    for param in layer.parameters():
      param.requires_grad = False

def freeze_layers(model):
    freeze(model.conv1)
    freeze(model.bn1)
    freeze(model.layer1)

    # freeze all BatchNorm2d layers
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            freeze(module)
    model.layer2.train()
    model.layer3.train()


def get_backbone(checkpoint_path):
    model = ResNet_imagenet(numberofclass=1000)     # must be same as used in checkpoint
    saved_checkpoint = torch.load(checkpoint_path, weights_only=True)
    if 'state_dict' in saved_checkpoint:
        saved_checkpoint = saved_checkpoint['state_dict']
    saved_checkpoint = {k.replace('module.', ''): v for k, v in saved_checkpoint.items()}
    model.load_state_dict(saved_checkpoint)
    freeze_layers(model)

    # remove avgpool & fc layers
    modules = [module for module in model.children() if not isinstance(module, nn.AvgPool2d) and not isinstance(module, nn.Linear)]
    backbone = torch.nn.Sequential(*modules)
    backbone.out_channels = 2048  # for ResNet-50, last layer out -> 512 x 4 = 2048
    return backbone


def get_faster_rcnn_model(checkpoint_path):
    backbone = get_backbone(checkpoint_path)

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=21,  # 20 classes + 1 for background
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    
    return model


def main(checkpoint_path: str, num_epochs=14, batch_size=8):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert os.path.exists(checkpoint_path), f"Checkpoint path '{checkpoint_path}' does not exist."

    dataset, dataset_test = get_voc_datasets()      # returns both 2007 & 2012

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size//2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    model = get_faster_rcnn_model(checkpoint_path).cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=4e-3, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    best_map = 0.0

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=500)

        scheduler.step()

        coco_eval = evaluate(model, data_loader_test, device=device)
        current_map = coco_eval.coco_eval['bbox'].stats[0]  # mAP (mean Average Precision)

        if current_map > best_map:
            best_map = current_map
            print(f"New best mAP: {100*best_map:.2f}, saving model...")
            date_suffix = get_current_timestamp_string()
            torch.save(model.state_dict(), f'best_fasterrcnn_model_{date_suffix}.pt')
            print(f"Best model saved as 'best_fasterrcnn_model_{date_suffix}.pt'.")

