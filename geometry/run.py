"""
Visualizing the geometry inside the concept embedding
The code is inspired from the prior work from "THE GEOMETRY OF CATEGORICAL AND HIERARCHICAL CONCEPTS IN LARGE LANGUAGE MODELS"
https://github.com/KihoPark/LLM_Categorical_Hierarchical_Representations
"""

import argparse
import json
import os
from pathlib import Path

import torch
import seaborn as sns

from load_rotre_embeddings import (
    load_embeddings, whiten_embeddings, load_entity_to_index, build_vocab_list
)
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from category import estimate_single_dir_from_embeddings
import plotting as rplot

def read_label_maps(rawdata_dir: str):
    """Read rawdata/animals.json and plants.json to build id->label mapping.
    Returns a dict mapping original id string -> label lowercased.
    """
    mapping = {}
    for fname in ['animals.json', 'plants.json']:
        path = Path(rawdata_dir) / fname
        if not path.exists():
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for cat, content in data.items():
            children = content.get('children', [])
            for obj in children:
                orig_id = str(obj.get('id'))
                label = obj.get('label')
                if orig_id and label:
                    mapping[orig_id] = label.lower()
    return mapping


def ids_from_raw_category(data, category_name):
    cats = data.get(category_name, {})
    children = cats.get('children', [])
    return [str(c['id']) for c in children if 'id' in c]


def main(args):
    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    ent2idx = load_entity_to_index(args.entity_to_id)

    emb = load_embeddings(ckpt, map_location='cpu')
    g, mean, inv_sqrt = whiten_embeddings(emb)
    vocab_list = build_vocab_list(emb.shape[0], ent2idx, id_to_label=read_label_maps(args.rawdata_dir))

    with open(os.path.join(args.rawdata_dir, 'animals.json'), 'r') as f:
        animals_json = json.load(f)
    with open(os.path.join(args.rawdata_dir, 'plants.json'), 'r') as f:
        plants_json = json.load(f)

    categories = ['mammal', 'bird', 'reptile', 'fish', 'amphibian']

    animals_indices = {}
    animals_ids = {}
    for cat in categories:
        ids = ids_from_raw_category(animals_json, cat)
        mapped = [ent2idx[s] for s in ids if s in ent2idx]
        animals_indices[cat] = mapped
        animals_ids[cat] = ids

    all_animals_ids = [s for cat in categories for s in ids_from_raw_category(animals_json, cat)]
    all_animals_mapped = [ent2idx[s] for s in all_animals_ids if s in ent2idx]
    animals_indices['animal'] = all_animals_mapped

    plant_ids_all = []
    for k, v in plants_json.items():
        for child in v.get('children', []):
            plant_ids_all.append(str(child.get('id')))
    plant_mapped = [ent2idx[s] for s in plant_ids_all if s in ent2idx]

    index_to_label = vocab_list

    dirs = {}
    for cat, idxs in animals_indices.items():
        if len(idxs) == 0:
            # no mapped entities 
            continue
        tensors = g[idxs]
        lda_dir, mean_dir = estimate_single_dir_from_embeddings(tensors)
        dirs[cat] = {'lda': lda_dir, 'mean': mean_dir}

    if len(plant_mapped) > 0:
        plant_tensors = g[plant_mapped]
        lda_dir_plant, mean_plant = estimate_single_dir_from_embeddings(plant_tensors)
        dirs_plants = {'lda': lda_dir_plant, 'mean': mean_plant}
    else:
        dirs_plants = None

    os.makedirs(args.output_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(25, 7))

    inds0 = {'animal': animals_indices['animal'], 'mammal': animals_indices.get('mammal', [])}
    dir1 = dirs['animal']['lda']
    dir2 = dirs['mammal']['lda']

    inds0 = {'animal': animals_indices['animal'], 'mammal': animals_indices.get('mammal', [])}
    inds1 = {'animal': animals_indices['animal'], 'mammal': animals_indices['mammal'], 'bird': animals_indices['bird']}
    inds2 = {'plant': plant_mapped, 'animal': animals_indices['animal'], 'mammal': animals_indices['mammal'], 'bird': animals_indices['bird']}

    unique_labels = list(dict.fromkeys(list(inds0.keys()) + list(inds1.keys()) + list(inds2.keys())))

    custom_colors = ['#f64369', '#2aab8c', '#48d1e8', '#5170ff']

    category_colors = {lab: custom_colors[i] for i, lab in enumerate(unique_labels)}

    rplot.proj_2d(dir1, dir2, g, index_to_label, axs[0], is_plain=True, double=False, higher1=None, subcat1=None, normalize=True, orthogonal=True,
                added_inds=inds0, category_colors=category_colors, k=200, fontsize=12, draw_arrows=True,
                arrow1_name='animal', arrow2_name='mammal', alpha=0.03, s=0.05,
                target_alpha=0.6, target_s=4, xlim=(-7, 7), ylim=(-7, 7),
                left_topk=False, bottom_topk=False, right_topk=False, top_topk=False,
                xlabel='', ylabel='', title='animal vs mammal')
    

    inds1 = {'animal': animals_indices['animal'], 'mammal': animals_indices['mammal'], 'bird': animals_indices['bird']}
    higher = dirs['animal']['lda']
    subcat1 = dirs['mammal']['lda']
    subcat2 = dirs['bird']['lda']

    rplot.proj_2d_single_diff(higher, subcat1, subcat2, g, index_to_label, axs[1], normalize=True, orthogonal=True,
                             added_inds=inds1, category_colors=category_colors, k=50, fontsize=12, draw_arrows=True,
                             arrow1_name='animal', arrow2_name='bird - mammal', alpha=0.03, s=0.05,
                             target_alpha=0.6, target_s=4, xlim=(-7, 7), ylim=(-7, 7), right_topk=False,
                             left_topk=False, top_topk=False, bottom_topk=False, xlabel='', ylabel='',
                             title='animal vs mammal => bird')

    inds2 = {'plant': plant_mapped, 'animal': animals_indices['animal'], 'mammal': animals_indices['mammal'], 'bird': animals_indices['bird']}
    higher1 = dirs_plants['lda'] if dirs_plants else None
    higher2 = dirs['animal']['lda']

    rplot.proj_2d_double_diff(higher1, higher2, subcat1, subcat2, g, index_to_label, axs[2], normalize=True,
                             orthogonal=True, added_inds=inds2, category_colors=category_colors, k=50, fontsize=12, draw_arrows=True,
                             arrow1_name='animal - plant', arrow2_name='bird - mammal', alpha=0.03, s=0.05,
                             target_alpha=0.6, target_s=4, xlim=(-7, 7), ylim=(-7, 7), right_topk=False,
                             left_topk=False, top_topk=False, bottom_topk=False, xlabel='', ylabel='',
                             title='plant -> animal vs mammal -> bird')

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'three_2d_plots_rotre.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 3D plots
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(121, projection='3d')

    cat1 = 'mammal'; cat2 = 'bird'; cat3 = 'fish'
    dir1 = dirs[cat1]['lda']; dir2 = dirs[cat2]['lda']; dir3 = dirs[cat3]['lda']
    higher_dir = dirs['animal']['lda']

    xaxis = dir1 / dir1.norm()
    yaxis = dir2 - (dir2 @ xaxis) * xaxis
    yaxis = yaxis / yaxis.norm()
    zaxis = dir3 - (dir3 @ xaxis) * xaxis - (dir3 @ yaxis) * yaxis
    zaxis = zaxis / zaxis.norm()
    axes = torch.stack([xaxis, yaxis, zaxis], dim=1)

    ind1 = animals_indices['mammal']
    ind2 = animals_indices['bird']
    ind3 = animals_indices['fish']

    g1 = g[ind1]
    g2 = g[ind2]
    g3 = g[ind3]

    proj1 = (g1 @ axes).cpu().numpy()
    proj2 = (g2 @ axes).cpu().numpy()
    proj3 = (g3 @ axes).cpu().numpy()
    proj = (g @ axes).cpu().numpy()

    P1 = (dir1 @ axes).cpu().numpy()
    P2 = (dir2 @ axes).cpu().numpy()
    P3 = (dir3 @ axes).cpu().numpy()
    P4 = (higher_dir @ axes).cpu().numpy()

    # scatter and arrows
    ax.scatter(P1[0], P1[1], P1[2], color='#f64369', s=100)
    ax.scatter(P2[0], P2[1], P2[2], color='#2aab8c', s=100)
    ax.scatter(P3[0], P3[1], P3[2], color='#48d1e8', s=100)

    verts = [list(zip([P1[0], P2[0], P3[0]], [P1[1], P2[1], P3[1]], [P1[2], P2[2], P3[2]]))]
    triangle = Poly3DCollection(verts, alpha=.2, linewidths=1, linestyle='--', edgecolors='#5170ff')
    triangle.set_facecolor('#ffb700')
    ax.add_collection3d(triangle)

    ax.quiver(0, 0, 0, P1[0], P1[1], P1[2], color='#f64369', arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P2[0], P2[1], P2[2], color='#2aab8c', arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P3[0], P3[1], P3[2], color='#48d1e8', arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P4[0], P4[1], P4[2], color='#5170ff', arrow_length_ratio=0.1, linewidth=2)

    ax.scatter(proj1[:, 0], proj1[:, 1], proj1[:, 2], c='#f64369', label=cat1)
    ax.scatter(proj2[:, 0], proj2[:, 1], proj2[:, 2], c='#2aab8c', label=cat2)
    ax.scatter(proj3[:, 0], proj3[:, 1], proj3[:, 2], c='#48d1e8', label=cat3)
    ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c='grey', s=0.05, alpha=0.03)


    scale = 1.2
    ax.text(P1[0]*scale + 2, P1[1]* scale, P1[2]*scale, cat1, bbox=dict(facecolor='#f64369', alpha=0.2))
    ax.text(P2[0]*scale+0.5, P2[1]* scale+0.5, P2[2]*scale, cat2, bbox=dict(facecolor='#2aab8c', alpha=0.2))
    ax.text(P3[0]*scale, P3[1]* scale, P3[2]*scale, cat3, bbox=dict(facecolor='#48d1e8', alpha=0.2))
    ax.text(P4[0]-0.6, P4[1]-0.6, P4[2], rf'$\bar{{\ell}}_{{animal}}$', bbox=dict(facecolor='#5170ff', alpha=0.2))

    ax.view_init(elev=20, azim=75)

    ax = fig.add_subplot(122, projection='3d')
    cat4 = 'reptile'
    dir4 = dirs[cat4]['lda']

    xaxis = (dir2 - dir1) / (dir2 - dir1).norm()
    yaxis = dir3 - dir1 - (dir3 - dir1) @ xaxis * xaxis
    yaxis = yaxis / yaxis.norm()
    zaxis = (dir4 - dir1) - (dir4 - dir1) @ xaxis * xaxis - (dir4 - dir1) @ yaxis * yaxis
    zaxis = zaxis / zaxis.norm()
    axes = torch.stack([xaxis, yaxis, zaxis], dim=1)

    ind4 = animals_indices['reptile']
    g4 = g[ind4]

    proj1 = (g[ind1] @ axes).cpu().numpy()
    proj2 = (g[ind2] @ axes).cpu().numpy()
    proj3 = (g[ind3] @ axes).cpu().numpy()
    proj4 = (g4 @ axes).cpu().numpy()
    proj = (g @ axes).cpu().numpy()

    P1 = (dir1 @ axes).cpu().numpy()
    P2 = (dir2 @ axes).cpu().numpy()
    P3 = (dir3 @ axes).cpu().numpy()
    P4 = (dir4 @ axes).cpu().numpy()

    ax.scatter(P1[0], P1[1], P1[2], color='#f64369', s=100)
    ax.scatter(P2[0], P2[1], P2[2], color='#2aab8c', s=100)
    ax.scatter(P3[0], P3[1], P3[2], color='#48d1e8', s=100)
    ax.scatter(P4[0], P4[1], P4[2], color='#5170ff', s=100)

    # some polygons
    verts1 = [list(zip([P1[0], P2[0], P3[0]], [P1[1], P2[1], P3[1]], [P1[2], P2[2], P3[2]]))]
    triangle1 = Poly3DCollection(verts1, alpha=.1, linewidths=1, linestyle='--', edgecolors='#5170ff')
    triangle1.set_facecolor('#ffb700')
    ax.add_collection3d(triangle1)

    verts2 = [list(zip([P1[0], P2[0], P4[0]], [P1[1], P2[1], P4[1]], [P1[2], P2[2], P4[2]]))]
    triangle2 = Poly3DCollection(verts2, alpha=.2, linewidths=1, linestyle='--', edgecolors='#5170ff')
    triangle2.set_facecolor('#ffb700')
    ax.add_collection3d(triangle2)

    verts3 = [list(zip([P1[0], P3[0], P4[0]], [P1[1], P3[1], P4[1]], [P1[2], P3[2], P4[2]]))]
    triangle3 = Poly3DCollection(verts3, alpha=.1, linewidths=1, linestyle =  "--", edgecolors='#5170ff')
    triangle3.set_facecolor('#ffb700')
    ax.add_collection3d(triangle3)

    verts4 = [list(zip([P2[0], P3[0], P4[0]], [P2[1], P3[1], P4[1]], [P2[2], P3[2], P4[2]]))]
    triangle4 = Poly3DCollection(verts4, alpha=.1, linewidths=1, linestyle =  "--", edgecolors='#5170ff')
    triangle4.set_facecolor('#ffb700')
    ax.add_collection3d(triangle4)

    ax.quiver(0, 0, 0, P1[0], P1[1], P1[2], color='#f64369', arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P2[0], P2[1], P2[2], color='#2aab8c', arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P3[0], P3[1], P3[2], color='#48d1e8', arrow_length_ratio=0.01)
    ax.quiver(0, 0, 0, P4[0], P4[1], P4[2], color='#5170ff', arrow_length_ratio=0.01)


    ax.scatter(proj1[:, 0], proj1[:, 1], proj1[:, 2], c='#f64369', label=cat1)
    ax.scatter(proj2[:, 0], proj2[:, 1], proj2[:, 2], c='#2aab8c', label=cat2)
    ax.scatter(proj3[:, 0], proj3[:, 1], proj3[:, 2], c='#48d1e8', label=cat3)
    ax.scatter(proj4[:, 0], proj4[:, 1], proj4[:, 2], c='#5170ff', label=cat4)
    ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c='gray', s=0.05, alpha=0.01)

    scale = 1.4
    scale2 = 1.2
    ax.text(P1[0]*scale-1, P1[1]* scale, P1[2]*scale, cat1, bbox=dict(facecolor='#f64369', alpha=0.2))
    ax.text(P2[0]*scale+1, P2[1]* scale, P2[2]*scale, cat2, bbox=dict(facecolor='#2aab8c', alpha=0.2))
    ax.text(P3[0]*scale-1, P3[1]* scale, P3[2]*scale, cat3, bbox=dict(facecolor='#48d1e8', alpha=0.2))
    ax.text(P4[0]*scale2+2, P4[1]* scale2, P4[2]*scale2-1, cat4, bbox=dict(facecolor='#5170ff', alpha=0.2))


    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'two_3d_plots_rotre.png'), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='RotE_model_20251201_144211_best.pt')
    parser.add_argument('--entity_to_id', type=str, default='UKC_CUT_1_hyp_t/entity_to_id.pickle')
    parser.add_argument('--rawdata_dir', type=str, default='rawdata')
    parser.add_argument('--output_dir', type=str, default='figures')
    args = parser.parse_args()

    main(args)
