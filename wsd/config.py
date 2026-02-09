"""Configuration for WSD training."""

# Data
# DATA_FILE = 'sentence_embeddings_labels.npz'
DATA_FILE = 'sentence_embeddings_labels_loc.npz'
# DATA_FILE = 'sentence_embeddings_labels_loc_xlm_roberta.npz'
# DATA_FILE = 'sentence_embeddings_labels_loc_distilbert_base_multilingual_cased.npz'
# DATA_FILE = 'sentence_embeddings_labels_loc_bert_base_uncased.npz'
# DATA_FILE = 'sentence_embeddings_labels_loc_roberta_base.npz'
# DATA_FILE = 'sentence_embeddings_qwen_labels.npz';
ROTE_MODEL = 'RotE_model_20251201_144211_best.pt'
TRAIN_TSV = 'dataset/train'
EVAL_TSV = 'dataset/eval'
TEST_TSV = 'dataset/test'

# Training
NUM_EPOCHS = 1
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9
OPTIM = 'Adam'
DEVICE = 'cuda'
TEMPERATURE = 0.1

# Model
INPUT_DIM = 768
HIDDEN_DIM = 1024
OUTPUT_DIM = 500
DROPOUT = 0.1
NUM_CONCEPTS = 37123

# for testing)
LIMIT_TRAIN_SAMPLES = None
MAX_SEQUENCE_LENGTH = 128

# Logging
LOG_INTERVAL = 10
USE_WANDB = True
WANDB_PROJECT = 'wsd-experiments'

# Output
OUTPUT_DIR = 'outputs'
MODEL_SAVE_PATH = 'concept_classifier.pth'
EVAL_PREDICTIONS_FILE = 'eval_predictions.txt'
TEST_PREDICTIONS_FILE = 'test_predictions.txt'
