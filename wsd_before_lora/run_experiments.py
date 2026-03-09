"""Script to run experiments with different hyperparameters."""

import itertools
from train import train

# Define hyperparameter search space
# HYPERPARAMETERS = {
#     'epochs': [5, 10, 15],
#     'batch_size': [16, 32],
#     'lr': [5e-6, 1e-5, 2e-5, 3e-5, 5e-5],
#     'weight_decay': [1e-5, 1e-4]
# }

# HYPERPARAMETERS = {
#     'epochs': [15],
#     'batch_size': [16, 32],
#     'lr': [2e-4, 5e-5, 1e-4, 1e-3, 1e-2, 2e-5],
#     'weight_decay': [1e-5, 0, 2e-2],
#     'do': [0.1],
#     'optim': ['Adafactor'],
#     'momentum': [0.9, 0.5],
#     'temperature': [1, 1.1, 1.3, 1.5],
#     'encoder': ['mmbert'],
# }
# # 5e-5, is pretty low

# HYPERPARAMETERS = {
#     'epochs': [15],
#     'batch_size': [16],
#     'lr': [2e-4, 1e-3, 2e-5],
#     'weight_decay': [1e-5, 2e-2],
#     'do': [0.1],
#     'optim': ['Adafactor'],
#     'momentum': [0.9, 0.5],
#     'temperature': [2, 1, 1.5],
#     'encoder': ['mmbert'],
# }


HYPERPARAMETERS = {
    'epochs': [15],
    'batch_size': [128, 32],
    'lr': [5e-5, 3e-5],
    'weight_decay': [5e-3, 2e-3],
    'do': [0.1],
    'optim': ['AdamW'],
    'momentum': [0.9],
    'temperature': [1],
    'encoder': ['mmbert'],
}


def run_experiments(param_grid=None):
    """Run training with different hyperparameter combinations."""
    
    # if param_grid is None:
    param_grid = HYPERPARAMETERS
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))

    print(combinations)
    
    print(f"Running {len(combinations)} experiments...\n")
    
    results = []
    for i, combo in enumerate(combinations, 1):
        config_dict = dict(zip(keys, combo))
        
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(combinations)}")
        print(f"Config: {config_dict}")
        print(f"{'='*60}")
        
        try:
            model, eval_preds, test_preds = train(config_dict)
            results.append({
                'config': config_dict,
                'status': 'success',
                'eval_samples': len(eval_preds),
                'test_samples': len(test_preds)
            })
            print(f"✓ Completed successfully")
        except Exception as e:
            results.append({
                'config': config_dict,
                'status': 'failed',
                'error': str(e)
            })
            print(f"✗ Failed: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")
    
    if failed:
        print("\nFailed experiments:")
        for r in failed:
            print(f"  {r['config']}: {r['error']}")
    
    return results


def run_limited_grid():
    """Run a smaller grid for quick testing."""
    param_grid = {
        'epochs': [1],
        'batch_size': [8],
        'lr': [1e-3],
        'weight_decay': [1e-5]
    }
    return run_experiments(HYPERPARAMETERS)


if __name__ == '__main__':
    # Run limited grid for testing
    # Change to run_experiments() for full grid
    results = run_limited_grid()
