"""Dataset classes and data loading utilities."""

import torch
import numpy as np
import pandas as pd
import ast
from torch.utils.data import Dataset


class WSDDataset(Dataset):
    def __init__(self, embeddings, labels, candidates, ids, answers=None):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = torch.from_numpy(labels).long()
        self.candidates = [np.array(c, dtype=np.int64) for c in candidates]
        self.ids = ids
        self.answers = answers
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'labels': self.labels[idx],
            'candidates': self.candidates[idx],
            'id': self.ids[idx],
            'answer': self.answers[idx] if self.answers else None
        }


def collate_wsd_batch(batch):
    return {
        'embedding': torch.stack([b['embedding'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'candidates': [b['candidates'] for b in batch],
        'id': [b['id'] for b in batch],
        'answer': [b['answer'] for b in batch]
    }
# 
# Loading RotE
def load_npz_data(npz_file):
    data = np.load(npz_file)
    return {
        'train_embeddings': data['train_embeddings'],
        'train_labels': data['train_labels'],
        'eval_embeddings': data['eval_embeddings'],
        'eval_labels': data['eval_labels'],
        'test_embeddings': data['test_embeddings'],
        'test_labels': data['test_labels']
    }


def load_metadata(train_tsv, eval_tsv, test_tsv):
    def parse_tsv(filepath):
        df = pd.read_csv(filepath, sep='\t')
        ids = df['id'].tolist()
        candidates = [ast.literal_eval(c) for c in df['candidates_id']]
        answers = df['answer_id'].tolist() if 'answer_id' in df.columns else None
        return ids, candidates, answers
    
    train_ids, train_cands, train_ans = parse_tsv(train_tsv)
    eval_ids, eval_cands, eval_ans = parse_tsv(eval_tsv)
    test_ids, test_cands, test_ans = parse_tsv(test_tsv)
    
    return {
        'train': (train_ids, train_cands, train_ans),
        'eval': (eval_ids, eval_cands, eval_ans),
        'test': (test_ids, test_cands, test_ans)
    }
