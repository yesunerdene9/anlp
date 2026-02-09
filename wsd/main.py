"""WSD training"""

import argparse
from train import train
from run_experiments import run_experiments


def main():
    parser = argparse.ArgumentParser(description='WSD Classification')
    parser.add_argument('--mode', choices=['train', 'experiments'], 
                       default='train', help='single train or experiment')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-2, help='Weigth decay')
    parser.add_argument('--do', type=float, default=0.1, help='Weigth decay')
    parser.add_argument('--temperature', type=float, default=1, help='Temperature')
    parser.add_argument('--optim', default="Adafactor", help='Optimizer')
    parser.add_argument('--momentum', default=0.9, help='momentum')
    parser.add_argument('--limit-samples', type=int, default=None, help='Limit training samples (for testing)')
    parser.add_argument('--desc', default="Training with concept embedding is trainable, with Adafactor and lr scheduler")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Mode: Single training run")
        config_dict = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'do': args.do,
            'weight_decay': args.wd,
            'optim': args.optim,
            'momentum': args.momentum,
            'temperature': args.temperature,
            'encoder': 'mmbert',
        }
        model, eval_preds, test_preds = train(config_dict)

    elif args.mode == 'experiments':
        print("Mode: Full experiments grid")
        results = run_experiments()
        
if __name__ == '__main__':
    main()
