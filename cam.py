import argparse
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
from const import imagenet_mean, imagenet_std


features = []
def hook_feature(module, input, output):
    features.append(output.detach())

def get_transformed_tensor_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0), np.array(img)


def generate_cam(feature_conv, weight_fc, class_idx):
    bz, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam


def overlay_cam_on_image(image_numpy, cam, output_path="cam_output.jpg"):
    cam = cv2.resize(cam, (image_numpy.shape[1], image_numpy.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img_bgr = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
    result = heatmap * 0.4 + img_bgr * 0.6
    
    cv2.imwrite(output_path, result)
    cv2.imshow('Class Activation Map', np.uint8(result))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(img_path, target_class=None):
    global features
    features = []

    img_tensor, image_numpy = get_transformed_tensor_image(img_path)

    model = models.resnet50(pretrained=True)
    model.eval()
    model.layer4.register_forward_hook(hook_feature)

    output = model(img_tensor)

    if target_class is None:
        target_class = output.argmax(dim=1).item()
        print(f"Predicted class index: {target_class}")

    fc_weight = model.fc.weight.data.cpu().numpy()
    last_conv_feature = features[0].cpu().numpy()

    cam = generate_cam(last_conv_feature, fc_weight, target_class)

    overlay_cam_on_image(image_numpy, cam)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Class Activation Map (CAM) for an image.")
    parser.add_argument("--image_path", type=str, help="Path to the input image.")
    parser.add_argument("--target_class", type=int, default=None, help="Target class index (optional).")
    args = parser.parse_args()
    
    main(args.image_path, args.target_class)