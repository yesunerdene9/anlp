"""Analysis utilities for WSD results."""

import numpy as np
from collections import Counter


def analyze_predictions(predictions, name="Predictions"):
    """Analyze prediction statistics."""
    
    if not predictions:
        print(f"{name}: No predictions")
        return
    
    concept_ids = [p[1] for p in predictions]
    
    print(f"\n{'='*60}")
    print(f"{name} Analysis")
    print(f"{'='*60}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Unique concepts: {len(set(concept_ids))}")
    print(f"Concept ID range: {min(concept_ids)} - {max(concept_ids)}")
    
    # Frequency analysis
    counter = Counter(concept_ids)
    most_common = counter.most_common(10)
    
    print(f"\nTop 10 Most Predicted Concepts:")
    for i, (concept_id, count) in enumerate(most_common, 1):
        print(f"  {i:2d}. Concept {concept_id}: {count} predictions ({count/len(predictions)*100:.2f}%)")
    
    # Distribution stats
    print(f"\nFrequency Statistics:")
    freqs = list(counter.values())
    print(f"  Mean frequency: {np.mean(freqs):.2f}")
    print(f"  Median frequency: {np.median(freqs):.2f}")
    print(f"  Std dev: {np.std(freqs):.2f}")
    print(f"  Min frequency: {min(freqs)}")
    print(f"  Max frequency: {max(freqs)}")


def compare_predictions(eval_pred, test_pred):
    """Compare eval and test predictions."""
    
    print(f"\n{'='*60}")
    print("Comparison: Eval vs Test")
    print(f"{'='*60}")
    
    eval_concepts = set(p[1] for p in eval_pred)
    test_concepts = set(p[1] for p in test_pred)
    
    print(f"Eval unique concepts: {len(eval_concepts)}")
    print(f"Test unique concepts: {len(test_concepts)}")
    print(f"Intersection: {len(eval_concepts & test_concepts)}")
    print(f"Only in Eval: {len(eval_concepts - test_concepts)}")
    print(f"Only in Test: {len(test_concepts - eval_concepts)}")


def load_predictions(filepath):
    """Load predictions from file."""
    predictions = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                sample_id = parts[0]
                concept_id = int(parts[1])
                predictions.append((sample_id, concept_id))
    return predictions


if __name__ == '__main__':
    # Example usage
    eval_pred = load_predictions('outputs/eval_predictions.txt')
    test_pred = load_predictions('outputs/test_predictions.txt')
    
    analyze_predictions(eval_pred, "Eval Predictions")
    analyze_predictions(test_pred, "Test Predictions")
    compare_predictions(eval_pred, test_pred)
