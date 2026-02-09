"""Plotting utilities"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


sns.set_theme(context="paper", style="white", palette="colorblind", font="DejaVu Sans", font_scale=2)


def proj_2d(dir1, dir2, unembed, vocab_list, ax, is_plain, double=False, higher1=None, subcat1=None,
            added_inds=None,
            category_colors=None,
            normalize=True,
            orthogonal=False, k=10, fontsize=12,
            alpha=0.2, s=0.5,
            target_alpha=0.9, target_s=2,
            xlim=None,
            ylim=None,
            draw_arrows=False,
            arrow1_name=None,
            arrow2_name=None,
            right_topk=True,
            left_topk=True,
            top_topk=True,
            bottom_topk=True,
            xlabel="dir1",
            ylabel="dir2",
            title="2D projection plot"):
    original_dir1 = dir1
    original_dir2 = dir2

    if normalize:
        dir1 = dir1 / dir1.norm()
        dir2 = dir2 / dir2.norm()
    if orthogonal:
        dir1 = dir1 / dir1.norm()
        dir2 = dir2 - (dir2 @ dir1) * dir1
        dir2 = dir2 / dir2.norm()

        arrow1 = [(original_dir1 @ dir1).cpu().numpy(), 0]
        arrow2 = [(original_dir2 @ dir1).cpu().numpy(), (original_dir2 @ dir2).cpu().numpy()]

    proj1 = unembed @ dir1
    proj2 = unembed @ dir2

    ax.scatter(proj1.cpu().numpy(), proj2.cpu().numpy(), alpha=alpha, color="gray", s=s)

    def _add_labels_for_largest(proj, largest):
        indices = torch.topk(proj, k=k, largest=largest).indices
        for idx in indices:
            if "$" not in vocab_list[idx]:
                ax.text(proj1[idx], proj2[idx], vocab_list[idx], fontsize=fontsize)

    if right_topk:
        _add_labels_for_largest(proj1, largest=True)
    if left_topk:
        _add_labels_for_largest(proj1, largest=False)
    if top_topk:
        _add_labels_for_largest(proj2, largest=True)
    if bottom_topk:
        _add_labels_for_largest(proj2, largest=False)

    if added_inds:
        colors = iter(["b", "r", "green", "orange", "skyblue", "pink",  "yellowgreen", "orange", "yellow", "brown", "cyan", "olive", "purple", "lime"])
        legend_handles = []
        for label, indices in added_inds.items():
            if category_colors is not None and label in category_colors:
                color = category_colors[label]
            else:
                color = next(colors)
            word_add = [vocab_list[i] for i in indices]
            for word, idx in zip(word_add, indices):
                ax.scatter(proj1[idx].cpu().numpy(), proj2[idx].cpu().numpy(), alpha=target_alpha, color=color, s=target_s)
            legend_handles.append(mpatches.Patch(color=color, label=label))
        ax.legend(handles=legend_handles, loc='lower left')

    if xlim is not None:
        ax.set_xlim(xlim)
        ax.hlines(0, xmax=xlim[1], xmin=xlim[0], colors="black", alpha=0.3, linestyles="dashed")
    else:
        ax.hlines(0, xmax=proj1.max().cpu().numpy(), xmin=proj1.min().cpu().numpy(), colors="black", alpha=0.3, linestyles="dashed")
    if ylim is not None:
        ax.set_ylim(ylim)
        ax.vlines(0, ymax=ylim[1], ymin=ylim[0], colors="black", alpha=0.3, linestyles="dashed")
    else:
        ax.vlines(0, ymax=proj2.max().cpu().numpy(), ymin=proj2.min().cpu().numpy(), colors="black", alpha=0.3, linestyles="dashed")


    if draw_arrows:
        if is_plain:
            ax.arrow(0, 0, arrow1[0], arrow1[1], head_width=0.5, head_length=0.5,
                    width=0.1, fc='blue', ec='blue',
                    linestyle='dashed',  alpha = 0.6, length_includes_head = True)
            if arrow1_name!=None:
                ax.text(arrow1[0]/2, arrow1[1]/2-1.5, arrow1_name, fontsize=fontsize,
                        bbox=dict(facecolor='blue', alpha=0.2))
            ax.arrow(0, 0, arrow2[0], arrow2[1], head_width=0.5, head_length=0.5,
                    width=0.1,  fc='red', ec='red',
                    linestyle='dashed',  alpha = 0.6, length_includes_head = True)
            if arrow2_name!=None:
                ax.text(arrow2[0]/2+1,
                        arrow2[1]/2, arrow2_name, fontsize=fontsize,
                        bbox=dict(facecolor='red', alpha=0.2))
        elif double:
            ax.arrow((higher1 @ dir1).cpu().numpy(),
                  (higher1 @ dir2).cpu().numpy(),
                  arrow1[0], arrow1[1], head_width=0.5, head_length=0.5,
                 width=0.1, fc='blue', ec='blue',
                 linestyle='dashed',  alpha = 0.6, length_includes_head = True)
            if arrow1_name!=None:
                ax.text((higher1 @ dir1).cpu().numpy()+ arrow1[0]*0.2, 
                        (higher1 @ dir2).cpu().numpy()+ arrow1[1]*0.2-1.5, arrow1_name, fontsize=fontsize,
                        bbox=dict(facecolor='blue', alpha=0.2))
            ax.arrow((subcat1 @ dir1).cpu().numpy(),
                    (subcat1 @ dir2).cpu().numpy(),
                    arrow2[0], arrow2[1], head_width=0.5, head_length=0.5,
                    width=0.1,  fc='red', ec='red',
                    linestyle='dashed',  alpha = 0.6, length_includes_head = True)
            if arrow2_name!=None:
                ax.text((subcat1 @ dir1).cpu().numpy()+ arrow2[0]/2+1,
                        (subcat1 @ dir2).cpu().numpy() + arrow2[1]/2, arrow2_name, fontsize=fontsize,
                        bbox=dict(facecolor='red', alpha=0.2))
        else:
            ax.arrow(0, 0, arrow1[0], arrow1[1], head_width=0.5, head_length=0.5,
                    width=0.1, fc='blue', ec='blue',
                    linestyle='dashed',  alpha = 0.6, length_includes_head = True)
            if arrow1_name!=None:
                ax.text(arrow1[0]/2, arrow1[1]/2-1.5, arrow1_name, fontsize=fontsize,
                        bbox=dict(facecolor='blue', alpha=0.2))
            ax.arrow((subcat1 @ dir1).cpu().numpy(),
                    (subcat1 @ dir2).cpu().numpy(),
                    arrow2[0], arrow2[1], head_width=0.5, head_length=0.5,
                    width=0.1,  fc='red', ec='red',
                    linestyle='dashed',  alpha = 0.6, length_includes_head = True)
            if arrow2_name!=None:
                ax.text((subcat1 @ dir1).cpu().numpy()+ 3*arrow2[0]/4-5,
                        (subcat1 @ dir2).cpu().numpy() + 3*arrow2[1]/4, arrow2_name, fontsize=fontsize,
                        bbox=dict(facecolor='red', alpha=0.2))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def proj_2d_single_diff(higher, subcat1, subcat2, unembed, vocab_list, ax, **kwargs):
    dir1 = higher
    dir2 = subcat2 - subcat1
    return proj_2d(dir1, dir2, unembed, vocab_list, ax, False, False, higher, subcat1, **kwargs)


def proj_2d_double_diff(higher1, higher2, subcat1, subcat2, unembed, vocab_list, ax, **kwargs):
    dir1 = higher2 - higher1
    dir2 = subcat2 - subcat1
    return proj_2d(dir1, dir2, unembed, vocab_list, ax, False, True, higher1, subcat1, **kwargs)
