import argparse
from detection.engine import evaluate
from detection import frcnn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object Detection Script')
    
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=14, help='Number of epochs to train')
    parser.add_argument('--backbone-checkpoint', type=str, default=None, required=True, help='Pre-trained backbone model weights to load')
    parser.add_argument('--frcnn-checkpoint', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model on the validation set')
    
    args = parser.parse_args()

    if args.evaluate:
        model = frcnn.get_faster_rcnn_model(args.backbone_checkpoint)
        dataset_test = frcnn.get_voc_datasets(test_only=True)

        coco_eval = evaluate(model, data_loader_test, device=device)
        current_map = coco_eval.coco_eval['bbox'].stats[0]
        print(f"Evaluation mAP: {100*current_map:.2f}")
    else:
        frcnn.main(args.backbone_checkpoint, num_epochs=args.epochs, batch_size=args.batch_size)
