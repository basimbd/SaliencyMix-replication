import os
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Class setup
CLASS_NAMES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS_NAME_TO_ID = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES) + 1


class VOCDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, voc_dataset, transforms=None):
        self.voc_dataset = voc_dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        img, target = self.voc_dataset[idx]
        orig_w, orig_h = img.size

        annots = target['annotation']
        boxes, labels = [], []

        objs = annots['object']
        if not isinstance(objs, list):
            objs = [objs]

        for obj in objs:
            bbox = obj['bndbox']
            xmin, ymin, xmax, ymax = map(float, (bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_NAME_TO_ID[obj['name']])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transforms:
            img = self.transforms(img)
            new_h, new_w = img.shape[1:]
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            boxes = boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y])

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        return img, target

    def __len__(self):
        return len(self.voc_dataset)


transform = T.Compose([
    T.Resize((480, 480)),
    T.ToTensor()
])

voc_2007_train = VOCDetection(root="./data", year="2007", image_set="trainval", download=True)
voc_2012_train = VOCDetection(root="./data", year="2012", image_set="trainval", download=True)

test_dataset = VOCDetection(root="./data", year="2007", image_set="test", download=True)

trainval_dataset = torch.utils.data.ConcatDataset([
    VOCDatasetWrapper(voc_2007_train, transforms=transform),
    VOCDatasetWrapper(voc_2012_train, transforms=transform)
])

train_loader = DataLoader(VOCDatasetWrapper(trainval_dataset, transforms=transform), batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(VOCDatasetWrapper(test_dataset, transforms=transform), batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = resnet_fpn_backbone('resnet50', pretrained=True)
model = FasterRCNN(backbone, num_classes=NUM_CLASSES)
model.to(device)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)


num_epochs = 20
best_loss = float('inf')
save_path = "./checkpoint/best_fasterrcnn_voc.pth"

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model at epoch {epoch+1}")


model.load_state_dict(torch.load(save_path))
model.eval()

def evaluate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            preds = [{"boxes": o["boxes"], "scores": o["scores"], "labels": o["labels"]} for o in outputs]
            targs = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
            metric.update(preds, targs)
    return metric.compute()

results = evaluate(model, test_loader, device)
print("Test Set mAP Results:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
