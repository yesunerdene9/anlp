"""Main training script with wandb logging."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import subprocess
from datetime import datetime
from tqdm import tqdm
import re

import config
from dataset import WSDDataset, load_npz_data, load_metadata, collate_wsd_batch
from models import ConceptClassifier
from utils import (load_rote_embeddings, create_bidirectional_mappings, 
                   load_concept_id_to_uk_id, evaluate_with_candidates, 
                   save_predictions, compute_metrics, create_concept_mask)
from types import SimpleNamespace

import wandb


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=True)
    for batch in pbar:
        embeddings = batch['embedding'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(embeddings)
        bs, sl = labels.shape
        
        # Reshape for loss computation
        logits_reshaped = logits.unsqueeze(1).expand(-1, sl, -1).reshape(-1, logits.shape[1])

        labels_reshaped = labels.reshape(-1)
        
        loss = loss_fn(logits_reshaped, labels_reshaped)

        loss = loss / sl
        
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # for name, p in model.named_parameters():
        #     if p.requires_grad:
        #         print(name, p.grad.abs().mean().item())
        #         break
                
        total_loss += loss.item() * bs
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader.dataset)


def evaluate(model, eval_loader, loss_fn, device, concept_id_to_index, index_to_concept_id):
    """Evaluate on eval/test set."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    
    pbar = tqdm(eval_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in pbar:
            embeddings = batch['embedding'].to(device)
            labels = batch['labels'].to(device)
            candidates = batch['candidates']
            ids = batch['id']
            
            logits = model(embeddings)
            bs, sl = labels.shape
            
            # Compute loss
            logits_reshaped = logits.unsqueeze(1).expand(-1, sl, -1).reshape(-1, logits.shape[1])
            labels_reshaped = labels.reshape(-1)
            loss = loss_fn(logits_reshaped, labels_reshaped)

            loss = loss / sl
            total_loss += loss.item() * bs
            
            # Get predictions with candidate limiting
            preds = evaluate_with_candidates(logits, candidates, labels, ids, 
                                            concept_id_to_index, index_to_concept_id, device)
            all_predictions.extend(preds)
    
    avg_loss = total_loss / len(eval_loader.dataset) if len(eval_loader.dataset) > 0 else 0
    return avg_loss, all_predictions


def compile_and_run_scorer():
    """Compile the Scorer.java file if needed."""
    scorer_java = 'Scorer.java'
    scorer_class = 'Scorer.class'
    
    if not os.path.exists(scorer_class):
        print("Scorer.java")
        result = subprocess.run(['javac', scorer_java], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed")
            return False
    return True


def evaluate_with_gold_standard(predictions_file, eval_dir, run_name):
    """Evaluate predictions against gold standard using Scorer.java"""
    gold_standards = [
        ('ALL', 'UKC.gold.key.txt'),
        ('Seen Only', 'UKC.in.test.gold.key.txt'),
        ('Unseen Only', 'UKC.out.test.gold.key.txt'),
        ('Single Candidate Only', 'UKC.single.test.gold.key.txt'),
        ('Multiple Candidates Only', 'UKC.multi.test.gold.key.txt'),
        ('Multiple Candidates Seen Only', 'UKC.multi.in.test.gold.key.txt'),
        ('Multiple Candidates Unseen Only', 'UKC.multi.out.test.gold.key.txt')
    ]
    
    predicted_ids = set()
    with open(predictions_file, 'r') as pf:
        for line in pf:
            parts = line.strip().split()
            if not parts:
                continue
            predicted_ids.add(parts[0])

    concept_id_to_index, _ = create_bidirectional_mappings()

    filtered_dir = os.path.join(os.path.dirname(predictions_file), 'filtered_gold')
    os.makedirs(filtered_dir, exist_ok=True)

    results = {}

    for title, gold_file in gold_standards:
        gold_path = os.path.join(eval_dir, gold_file)
        if not os.path.exists(gold_path):
            continue

        filtered_lines = []
        total_gold_lines = 0
        with open(gold_path, 'r') as gf:
            for line in gf:
                line = line.strip()
                if not line:
                    continue
                total_gold_lines += 1
                parts = line.split()
                gid = parts[0]
                gold_concepts = []
                for tok in parts[1:]:
                    try:
                        gold_concepts.append(int(tok))
                    except Exception:
                        pass

                if gid in predicted_ids and any((gc in concept_id_to_index) for gc in gold_concepts):
                    filtered_lines.append(line)

        filtered_path = os.path.join(filtered_dir, gold_file)
        with open(filtered_path, 'w') as outf:
            for l in filtered_lines:
                outf.write(l + '\n')

        effective_size = len(filtered_lines)
        coverage_gold = (effective_size / total_gold_lines * 100.0) if total_gold_lines > 0 else 0.0
        coverage_pred = (effective_size / len(predicted_ids) * 100.0) if len(predicted_ids) > 0 else 0.0

        try:
            result = subprocess.run(
                ['java', 'Scorer', filtered_path, predictions_file],
                capture_output=True, text=True, timeout=30
            )

            metrics = {}
            if result.returncode == 0:
                output = result.stdout
                p_match = re.search(r'P=\s*([\d.]+)%', output)
                r_match = re.search(r'R=\s*([\d.]+)%', output)
                f1_match = re.search(r'F1=\s*([\d.]+)%', output)
                if p_match and r_match and f1_match:
                    metrics = {
                        'precision': float(p_match.group(1)),
                        'recall': float(r_match.group(1)),
                        'f1': float(f1_match.group(1)),
                        'effective_size': effective_size,
                        'coverage_gold_pct': coverage_gold,
                        'coverage_predicted_pct': coverage_pred,
                        'gold_total': total_gold_lines
                    }
            results[title] = metrics
        except Exception as e:
            print(f"Failed {title}: {e}")

    return results


def train(config_dict=None):
    """Main training function."""
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if config_dict is None:
        config_dict = {}
    
    default_config = {
        'epochs': config.NUM_EPOCHS,
        'batch_size': config.BATCH_SIZE,
        'lr': config.LEARNING_RATE,
        'weight_decay': config.WEIGHT_DECAY,
        'temperature': config.TEMPERATURE,
        'momentum': config.MOMENTUM,
        'optim': config.OPTIM,
        'do': config.DROPOUT,
        'encoder': 'mmbert'
    }
    
    # Merge with provided config
    for key in default_config:
        if key not in config_dict:
            config_dict[key] = default_config[key]
    
    cf = SimpleNamespace(**config_dict)
    run_name = f"{run_timestamp}_lr{cf.lr}_wd{cf.weight_decay}_bt{cf.batch_size}_t{cf.temperature}_m{cf.momentum}_{cf.optim}_{cf.encoder}"
    
    output_base = config.OUTPUT_DIR
    run_output_dir = os.path.join(output_base, run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    wandb.init(project=config.WANDB_PROJECT, name=run_name, config=cf.__dict__)
    
    device = torch.device(config.DEVICE)

    compile_and_run_scorer()

    concept_id_to_index, index_to_concept_id = create_bidirectional_mappings()
    concept_id_to_uk_id = load_concept_id_to_uk_id()

    data = load_npz_data(config.DATA_FILE)
    metadata = load_metadata(config.TRAIN_TSV, config.EVAL_TSV, config.TEST_TSV)
    
    train_ids, train_cands, train_ans = metadata['train']
    eval_ids, eval_cands, eval_ans = metadata['eval']
    test_ids, test_cands, test_ans = metadata['test']
    
    if config.LIMIT_TRAIN_SAMPLES:
        data['train_embeddings'] = data['train_embeddings'][:config.LIMIT_TRAIN_SAMPLES]
        data['train_labels'] = data['train_labels'][:config.LIMIT_TRAIN_SAMPLES]
        train_ids = train_ids[:config.LIMIT_TRAIN_SAMPLES]
        train_cands = train_cands[:config.LIMIT_TRAIN_SAMPLES]
        train_ans = train_ans[:config.LIMIT_TRAIN_SAMPLES] if train_ans else None
    
    print(f"Data shapes: Train {data['train_embeddings'].shape}, "
          f"Eval {data['eval_embeddings'].shape}, Test {data['test_embeddings'].shape}")
    
    train_ds = WSDDataset(data['train_embeddings'], data['train_labels'], 
                          train_cands, train_ids, train_ans)
    eval_ds = WSDDataset(data['eval_embeddings'], data['eval_labels'], 
                         eval_cands, eval_ids, eval_ans)
    test_ds = WSDDataset(data['test_embeddings'], data['test_labels'], 
                         test_cands, test_ids, test_ans)
    
    train_loader = DataLoader(train_ds, batch_size=cf.batch_size, 
                             shuffle=True, collate_fn=collate_wsd_batch)
    eval_loader = DataLoader(eval_ds, batch_size=config.EVAL_BATCH_SIZE, 
                            shuffle=False, collate_fn=collate_wsd_batch)
    test_loader = DataLoader(test_ds, batch_size=config.EVAL_BATCH_SIZE, 
                            shuffle=False, collate_fn=collate_wsd_batch)
    
    print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}, "
          f"Test batches: {len(test_loader)}")
    
    rote_emb = load_rote_embeddings(config.ROTE_MODEL)
    
    concept_mask = create_concept_mask(train_ans, config.NUM_CONCEPTS, concept_id_to_index)
    concept_mask = concept_mask.to(device)

    model = ConceptClassifier(rote_emb, input_dim=config.INPUT_DIM, dropout=cf.do, 
                             temperature=cf.temperature, concept_mask=concept_mask).to(device)

    print("model:")
    print(model)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    
    print("Config\n")
    print(cf)

    if cf.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    elif cf.optim == 'Adafactor':
        optimizer = optim.Adafactor(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    elif cf.optim == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    elif cf.optim == 'RAdam':
        optimizer = optim.RAdam(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    elif cf.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay, momentum=cf.momentum)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)

    print(optimizer)


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
    
    print("\nStarting training")
    for epoch in range(1, cf.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        eval_loss, eval_preds = evaluate(model, eval_loader, loss_fn, device, 
                                        concept_id_to_index, index_to_concept_id)
        test_loss, test_preds = evaluate(model, test_loader, loss_fn, device,
                                        concept_id_to_index, index_to_concept_id)
        
        
        print(f"Epoch {epoch}/{cf.epochs}: "
              f"Train Loss={train_loss:.4f}, Eval Loss={eval_loss:.4f}, "
              f"Test Loss={test_loss:.4f}")
        
        current_lr = optimizer.param_groups[0]['lr']
    
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'test_loss': test_loss,
            'lr': current_lr,
        })
    
        eval_pred_file = os.path.join(run_output_dir, config.EVAL_PREDICTIONS_FILE)
        test_pred_file = os.path.join(run_output_dir, config.TEST_PREDICTIONS_FILE)
        model_file = os.path.join(run_output_dir, config.MODEL_SAVE_PATH)
        
        save_predictions(eval_preds, eval_pred_file, 
                        concept_id_to_uk_id)
        save_predictions(test_preds, test_pred_file,
                        concept_id_to_uk_id)

        print("\nEvaluating")
        eval_scores = evaluate_with_gold_standard(eval_pred_file, 'eval-gold-standard', 'Eval Set')
        test_scores = evaluate_with_gold_standard(test_pred_file, 'test-gold-standard', 'Test Set')
        
        eval_f1 = eval_scores.get('ALL', {}).get('f1', 0.0)
        scheduler.step(-eval_f1)
        
        for category, metrics in eval_scores.items():
            if category == 'Single Candidate Only':
                continue
            if not metrics:
                continue
            wandb.log({
                f'eval_precision/{category}': metrics.get('precision', 0.0),
                f'eval_recall/{category}': metrics.get('recall', 0.0),
                f'eval_f1/{category}': metrics.get('f1', 0.0),
            })

        for category, metrics in test_scores.items():
            if not metrics:
                continue
            wandb.log({
                f'test_precision/{category}': metrics.get('precision', 0.0),
                f'test_recall/{category}': metrics.get('recall', 0.0),
                f'test_f1/{category}': metrics.get('f1', 0.0),
            })

        # report table
        report_file = os.path.join(run_output_dir, 'evaluation_report.txt')
        with open(report_file, 'w') as rf:
            def write_table(title, scores):
                cols = [
                    ("Subset", 35),
                    ("Gold", 8),
                    ("Evaluated", 10),
                    ("Coverage (%)", 14),
                    ("Precision", 10),
                    ("Recall", 8),
                    ("F1", 6),
                ]

                def sep():
                    return "+" + "+".join("-" * w for _, w in cols) + "+\n"

                def row(values):
                    return "|" + "|".join(
                        f"{str(v):<{w}}" for v, (_, w) in zip(values, cols)
                    ) + "|\n"

                rf.write(title + "\n")
                rf.write(sep())
                rf.write(row([c[0] for c in cols]))
                rf.write(sep())

                print("\n" + title)
                print(sep().strip())
                print(row([c[0] for c in cols]).strip())
                print(sep().strip())

                subsets = [
                    "ALL",
                    "Seen Only",
                    "Unseen Only",
                    "Single Candidate Only",
                    "Multiple Candidates Only",
                    "Multiple Candidates Seen Only",
                    "Multiple Candidates Unseen Only",
                ]

                for subset in subsets:
                    metrics = scores.get(subset, {})
                    row_vals = [
                        subset,
                        int(metrics.get("gold_total", 0)),
                        int(metrics.get("effective_size", 0)),
                        f"{float(metrics.get('coverage_gold_pct', 0.0)):.1f}",
                        f"{float(metrics.get('precision', 0.0)):.1f}",
                        f"{float(metrics.get('recall', 0.0)):.1f}",
                        f"{float(metrics.get('f1', 0.0)):.1f}",
                    ]
                    rf.write(row(row_vals))
                    print(row(row_vals).strip())

                rf.write(sep())
                print(sep().strip())


            write_table('Eval Set Coverage & Scores', eval_scores)
            rf.write('\n')
            write_table('Test Set Coverage & Scores', test_scores)

    wandb.finish()
    
    return model, eval_preds, test_preds


if __name__ == '__main__':
    train()
