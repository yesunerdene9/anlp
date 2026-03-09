"""Utility functions for training and evaluation."""

import torch
import numpy as np
import pickle
import pandas as pd

import torch

def load_rote_embeddings(model_path):
    """Load RotE concept embeddings."""
    ckpt = torch.load(model_path, map_location='cpu')
    embeddings = ckpt['entity.weight'].detach().cpu().float()
    return embeddings


def load_entity_to_idx(pickle_path='dataset/entity_to_id.pickle'):
    """Load entity ID to index mapping.
    
    Returns:
        concept_id_to_index: dict mapping concept_id to embedding matrix index
    """
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def create_bidirectional_mappings(pickle_path='dataset/entity_to_id.pickle'):
    """Create bidirectional mappings between concept IDs and indices.
    
    Returns:
        concept_id_to_index: dict mapping concept_id (int) → embedding_index (int)
        index_to_concept_id: dict mapping embedding_index (int) → concept_id (int)
    """
    entity_to_idx = load_entity_to_idx(pickle_path)
    # Convert string keys to int
    concept_id_to_index = {int(cid_str): idx for cid_str, idx in entity_to_idx.items()}
    index_to_concept_id = {idx: cid for cid, idx in concept_id_to_index.items()}
    return concept_id_to_index, index_to_concept_id


def load_concept_id_to_uk_id(csv_path='dataset/concepts.csv'):
    """Load mapping from concept ID to uk_id.
    
    Returns:
        concept_id_to_uk_id: dict mapping concept_id → uk_id
    """
    df = pd.read_csv(csv_path)
    return dict(zip(df['id'], df['uk_id']))


def convert_candidates_to_indices(candidates, concept_id_to_index):
    """Convert candidate concept IDs to embedding matrix indices.
    
    Args:
        candidates: numpy array of concept IDs
        concept_id_to_index: dict mapping concept_id → embedding_index
    
    Returns:
        numpy array of indices
    """
    indices = []
    for cid in candidates:
        if cid in concept_id_to_index:
            indices.append(concept_id_to_index[cid])
    return np.array(indices, dtype=np.int64) if indices else np.array([0], dtype=np.int64)


def evaluate_with_candidates(logits, candidates, labels, ids, concept_id_to_index, 
                             index_to_concept_id, device='cpu'):
    """
    Evaluate predictions restricted to candidate concepts.
    
    Args:
        logits: (batch_size, num_concepts) or (batch_size, seq_len, num_concepts)
        candidates: list of numpy arrays with candidate concept IDs
        labels: (batch_size, seq_len) torch tensor (already indices)
        ids: list of sample IDs
        concept_id_to_index: dict mapping concept_id → embedding_index
        index_to_concept_id: dict mapping embedding_index → concept_id
        device: torch device
    
    Returns:
        predictions: list of (id, predicted_concept_id) tuples
    """
    predictions = []
    
    for i in range(len(candidates)):
        cand_ids = candidates[i]  # numpy array of concept IDs
        label_seq = labels[i].cpu().numpy() if isinstance(labels[i], torch.Tensor) else labels[i]
        
        # Find positions with valid labels
        valid_pos = np.where(label_seq != -100)[0]
        if len(valid_pos) == 0:
            continue
        
        pos = valid_pos[0]
        logit = logits[i, pos] if logits.dim() == 3 else logits[i]
        
        # Convert candidate concept IDs to indices
        cand_indices = convert_candidates_to_indices(cand_ids, concept_id_to_index)
        
        # Filter candidates to valid range
        max_idx = logit.shape[0]
        valid_cand_indices = cand_indices[cand_indices < max_idx]
        if len(valid_cand_indices) == 0:
            valid_cand_indices = np.array([0])
        
        # Get prediction from candidate-restricted logits
        cand_logits = logit[valid_cand_indices]
        pred_idx_in_candidates = cand_logits.argmax().item()
        pred_embedding_index = int(valid_cand_indices[pred_idx_in_candidates])
        
        # Convert embedding index back to concept ID
        pred_concept_id = index_to_concept_id.get(pred_embedding_index, 1)
        
        predictions.append((ids[i], pred_concept_id))
    
    return predictions


def save_predictions(predictions, filepath, concept_id_to_uk_id=None):
    """Save predictions in SemEval format: id uk_id.
    
    Args:
        predictions: list of (sample_id, concept_id) tuples
        filepath: output file path
        concept_id_to_uk_id: dict mapping concept_id → uk_id (optional)
    """
    with open(filepath, 'w') as f:
        for sample_id, concept_id in predictions:
            # Convert concept_id to uk_id if mapping provided
            if concept_id_to_uk_id and concept_id in concept_id_to_uk_id:
                output_id = concept_id_to_uk_id[concept_id]
            else:
                output_id = concept_id
            f.write(f"{sample_id} {output_id}\n")


def compute_metrics(logits, labels, ignore_index=-100):
    """Compute basic accuracy metrics."""
    if logits.dim() == 3:
        batch_size, seq_len, num_concepts = logits.shape
        logits = logits.reshape(-1, num_concepts)
    if labels.dim() == 2:
        labels = labels.reshape(-1)
    
    predictions = logits.argmax(dim=1)
    mask = labels != ignore_index
    
    if mask.sum() == 0:
        return {'accuracy': 0.0, 'num_valid': 0}
    
    correct = (predictions[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    
    return {
        'accuracy': correct / total if total > 0 else 0.0,
        'num_valid': total
    }


def create_concept_mask(train_answer_ids, num_concepts, concept_id_to_index):
    """Create a binary mask vector for concepts that exist in training set.
    
    Args:
        train_answer_ids: list of answer_id (concept IDs) from training set
        num_concepts: total number of concept embedding indices (e.g., 37123)
        concept_id_to_index: dict mapping concept_id → embedding_index
    
    Returns:
        mask: (num_concepts,) torch tensor with 1 for concepts in training set, 0 otherwise
    """
    mask = torch.zeros(num_concepts, dtype=torch.float32)
    
    # Set 1 for concepts that appear as answers in training set
    for concept_id in train_answer_ids:
        if concept_id in concept_id_to_index:
            embedding_idx = concept_id_to_index[concept_id]
            mask[embedding_idx] = 1.0
    
    return mask
