import argparse
from detection.engine import evaluate
from detection import frcnn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object Detection Script')
    
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=14, help='Number of epochs to train')
    parser.add_argument('--backbone-checkpoint', type=str, default=None, required=True, help='Pre-trained backbone model weights to load')
    parser.add_argument('--frcnn-checkpoint', type=int, default=4, help='Number of workers for data loading')
    
    args = parser.parse_args()

    frcnn.main(args.backbone_checkpoint, num_epochs=args.epochs, batch_size=args.batch_size)
