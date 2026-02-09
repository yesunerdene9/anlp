import torch
from sklearn.covariance import ledoit_wolf


def category_to_indices(category, id_map):
    return [id_map[w] for w in category if w in id_map]

def estimate_single_dir_from_embeddings(category_embeddings: torch.Tensor):
    if not isinstance(category_embeddings, torch.Tensor):
        category_embeddings = torch.tensor(category_embeddings)

    category_mean = category_embeddings.mean(dim=0)

    cov = ledoit_wolf(category_embeddings.cpu().numpy())
    cov = torch.tensor(cov[0], device=category_embeddings.device, dtype=category_embeddings.dtype)

    pseudo_inv = torch.linalg.pinv(cov)
    lda_dir = pseudo_inv @ category_mean
    lda_dir = lda_dir / torch.norm(lda_dir)
    lda_dir = (category_mean @ lda_dir) * lda_dir

    return lda_dir, category_mean
