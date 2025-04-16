import argparse
from detection.engine import evaluate
from detection import frcnn
import torch
from torch.utils.data import DataLoader
from detection import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object Detection Script')
    
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=14, help='Number of epochs to train')
    parser.add_argument('--backbone-checkpoint', type=str, default=None, required=True, help='Pre-trained backbone model weights to load')
    parser.add_argument('--frcnn-checkpoint', type=str, default=None, help='Faster R-CNN model weights to load')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model on the validation set')
    
    args = parser.parse_args()

    if args.evaluate:
        model = frcnn.get_faster_rcnn_model(args.backbone_checkpoint)
        if args.frcnn_checkpoint:
            checkpoint = torch.load(args.frcnn_checkpoint, weights_only=True)
            if 'model_state_dict' in checkpoint:
                checkpoint = checkpoint['model_state_dict']
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
            print(f"Loaded model from {args.frcnn_checkpoint}")
        model = model.cuda()
        dataset_test = frcnn.get_voc_datasets(test_only=True)
        data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size//2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

        coco_eval = evaluate(model, data_loader_test, device='cuda')
        current_map = coco_eval.coco_eval['bbox'].stats[0]
        print(f"Evaluation mAP: {100*current_map:.2f}")
    else:
        frcnn.main(args.backbone_checkpoint, num_epochs=args.epochs, batch_size=args.batch_size)
