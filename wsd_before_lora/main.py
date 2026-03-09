"""Main entry point for WSD classification."""

import argparse
import sys
import config
from train import train
from run_experiments import run_experiments, run_limited_grid
from visualize import save_plots
from analyze import analyze_predictions, compare_predictions, load_predictions


def main():
    parser = argparse.ArgumentParser(description='WSD Classification')
    parser.add_argument('--mode', choices=['train', 'experiments', 'quick', 'analyze'], 
                       default='train', help='Mode to run')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    # parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-3, help='Weigth decay')
    parser.add_argument('--do', type=float, default=0.1, help='Dropout')
    parser.add_argument('--temperature', type=float, default=1, help='Temperature')
    parser.add_argument('--optim', default="AdamW", help='Optimizer')
    parser.add_argument('--momentum', default=0.9, help='momentum')
    parser.add_argument('--limit-samples', type=int, default=None, help='Limit training samples (for testing)')
    parser.add_argument('--desc', default="Training with concept embedding is trainable, with adafactor and lr scheduler")
    
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
        save_plots(eval_preds, test_preds)
        
    elif args.mode == 'quick':
        print("Mode: Quick test (1 epoch, limited samples)")
        import config
        config.LIMIT_TRAIN_SAMPLES = 10
        config.NUM_EPOCHS = 1
        config_dict = {
            'epochs': 10,
            'batch_size': 32,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'temperature': 0.1
        }
        model, eval_preds, test_preds = train(config_dict)
        save_plots(eval_preds, test_preds)
        
    elif args.mode == 'experiments':
        print("Mode: Full experiments grid")
        results = run_experiments()
        
    elif args.mode == 'analyze':
        print("Mode: Analyze results")
        try:
            eval_pred = load_predictions('outputs/eval_predictions.txt')
            test_pred = load_predictions('outputs/test_predictions.txt')
            analyze_predictions(eval_pred, "Eval Predictions")
            analyze_predictions(test_pred, "Test Predictions")
            compare_predictions(eval_pred, test_pred)
        except FileNotFoundError:
            print("Error: Prediction files not found. Run training first.")
            sys.exit(1)


if __name__ == '__main__':
    main()
