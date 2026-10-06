import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from itertools import combinations
from scipy.optimize import linear_sum_assignment
from scipy.stats import hypergeom
#%% import some connectome data 
sensory_neurons = pd.read_csv("init_data/sensory_neurons.csv")
sensory_neurons['skids'] = sensory_neurons['skids'].apply(ast.literal_eval)

firstorder = sensory_neurons[['modality', 'skids']].explode('skids')
skid_to_modality = dict(zip(firstorder['skids'], firstorder['modality']))

con = pd.read_csv('init_data/connector_details2025.csv')
con['postsynaptic_to'] = con['postsynaptic_to'].apply(ast.literal_eval)
con['presynaptic_to'] = con['presynaptic_to'].astype(int)

#%%
d = pd.read_csv('data_mohammad/202609_fullmodel/details_with_labels.csv')
d.drop(columns=["Unnamed: 0"], inplace=True)

chains = ['chain25', 'chain39', 'chain83'] # hyperblock chains in data
# chains are: best LL, best ARI, best AMI

ct_names = d['celltype'].unique()
pred_nt = d['predicted_neurotransmitter'].unique()

print('K = 17')
print(f'Effective dimensionality: \n {[d[col].nunique() for col in chains]}')

#%% get effective dimensionality of each chain with numbers 

print('K = 17')
print(f'Effective dimensionality: \n {[d[col].nunique() for col in chains]}')

thresh_neurons = 3
chain_dims = {col: d[col].nunique() for col in chains}
chain_dims_over_threshold = {col: (d[col].value_counts() > thresh_neurons).sum() for col in chains}

fig, ax = plt.subplots(1, 2, figsize=(6, 4))
ax[0].bar(chain_dims.keys(), chain_dims.values())
ax[0].set_title('Effective Dimensionality')
ax[0].set_ylabel('Number of Blocks')
ax[0].set_ylim(0, 17.1)
ax[1].bar(chain_dims_over_threshold.keys(), chain_dims_over_threshold.values())
ax[1].set_title(f'Blocks Over Threshold ({thresh_neurons} Neurons)')
ax[1].set_ylabel('Number of Blocks')
ax[1].set_ylim(0, 17.1)
fig.tight_layout()


#%% 
def create_confusion_matrix(data, rows='celltype', cols='predicted_block'):
    confusion_matrix = pd.crosstab(data[rows], data[cols])
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, title='Confusion Matrix', fmt='d',
                          cmap='Purples', vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt=fmt, cmap=cmap, ax=ax, 
                vmin=vmin, vmax=vmax)
    ax.set_title(title)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)

fig, ax = plt.subplots(1, 3, figsize=(17, 7))
colormax = 355
for e, chain in enumerate(chains):
    cm = create_confusion_matrix(d, rows='celltype', cols=chain)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax[e], 
                vmin=None, vmax=colormax)
    ax[e].set_title(f'Confusion Matrix for {chain}')
    for _, spine in ax[e].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
for a in ax[1:]:
    a.set_yticks([])
    a.set_ylabel('')
for a in ax[:-1]:
    a.collections[0].colorbar.remove()
#fig.tight_layout()

# %% block matrching
for c1, c2 in [(chains[0], chains[1]), (chains[0], chains[2]), (chains[1], chains[2])]:
    cmblock = create_confusion_matrix(d, rows=c1, cols=c2)
    plot_confusion_matrix(cmblock, title=f'Block Matching Between {c1} and {c2}')

# %% block matching, split cells: row-normalised vs column-normalised
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

def plot_split_confusion_matrix(counts, title='Confusion Matrix',
                                cmap='Purples', annot_thresh=0.01, cell=0.32):
    """Each cell is cut along its top-left -> bottom-right diagonal.

    lower-left  triangle: row-normalised    counts / row total  -> P(col | row)
    upper-right triangle: column-normalised counts / col total  -> P(row | col)

    Both halves share one 0-1 scale and one colormap, so a cell dark on both
    sides is a genuine one-to-one block correspondence, while a cell dark on one
    side only is an asymmetric match (a small block swallowed by a large one).
    Annotations are fractions, drawn only above `annot_thresh` so the near-empty
    majority of cells stays clean.
    """
    n_r, n_c = counts.shape
    # replace(0, nan) keeps empty blocks from turning into inf rather than 0
    row_norm = counts.div(counts.sum(1).replace(0, np.nan), axis=0).fillna(0).to_numpy()
    col_norm = counts.div(counts.sum(0).replace(0, np.nan), axis=1).fillna(0).to_numpy()

    # cell (i, j) spans x in [j, j+1], y in [i, i+1]; y is inverted below so
    # row 0 sits at the top, matching seaborn/imshow orientation
    lower, upper = [], []
    for i in range(n_r):
        for j in range(n_c):
            tl, tr = (j, i), (j + 1, i)
            bl, br = (j, i + 1), (j + 1, i + 1)
            lower.append([tl, bl, br])  # below the tl->br diagonal
            upper.append([tl, tr, br])  # above it

    fig, ax = plt.subplots(figsize=(cell * n_c + 2.2, cell * n_r + 1.4))
    norm = Normalize(0, 1)
    for verts, values in ((lower, row_norm), (upper, col_norm)):
        coll = PolyCollection(verts, cmap=cmap, norm=norm,
                              edgecolors='white', linewidths=0.4)
        coll.set_array(values.ravel())
        ax.add_collection(coll)

    fontsize = max(3.5, min(7.0, 190 / max(n_r, n_c)))
    for i in range(n_r):
        for j in range(n_c):
            # triangle centroids, so the two numbers never overlap
            for value, (x, y) in ((row_norm[i, j], (j + 1 / 3, i + 2 / 3)),
                                  (col_norm[i, j], (j + 2 / 3, i + 1 / 3))):
                if value < annot_thresh:
                    continue
                # drop the leading zero: ".87" fits a triangle, "0.87" does not
                ax.text(x, y, f'{value:.2f}'.lstrip('0'), ha='center', va='center',
                        fontsize=fontsize,
                        color='white' if value > 0.6 else 'black')

    ax.set_xlim(0, n_c)
    ax.set_ylim(n_r, 0)  # inverted: first row on top
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(n_c) + 0.5)
    ax.set_yticks(np.arange(n_r) + 0.5)
    ax.set_xticklabels(counts.columns, rotation=90, fontsize=fontsize + 1)
    ax.set_yticklabels(counts.index, rotation=0, fontsize=fontsize + 1)
    ax.set_xlabel(counts.columns.name)
    ax.set_ylabel(counts.index.name)
    ax.tick_params(length=0)
    ax.set_title(title)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)

    # both halves share `norm`, so one bar reads for both triangles
    cb = fig.colorbar(coll, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('lower left: P(col | row)   upper right: P(row | col)', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig, ax


for c1, c2 in [(chains[0], chains[1]), (chains[0], chains[2]), (chains[1], chains[2])]:
    cmblock = create_confusion_matrix(d, rows=c1, cols=c2)
    plot_split_confusion_matrix(cmblock, title=f'Block Matching Between {c1} and {c2}')

# %% overlap metrics between the blocks of two chains
# Rows of every matrix below are blocks of chain `a`, columns are blocks of chain `b`.
# Each neuron appears once in every chain column, so correspondence is just row overlap.

TAU = 0.6    # containment above this counts as a "dominant" correspondence
SIGMA = 0.2  # minimum containment/jaccard worth reporting at all

chain_pairs = list(combinations(chains, 2))


def block_overlap_metrics(data, a, b, min_size=thresh_neurons):
    """Overlap metrics between blocks of chain column `a` and chain column `b`.

    Blocks with fewer than `min_size` neurons are dropped: they cannot support a
    meaningful match and otherwise dominate the small-denominator metrics.

    jaccard   - symmetric overlap, penalises size mismatch
    fwd       - containment A->B, fraction of block A that lands in block B
    bwd       - containment B->A, fraction of block B that came from block A
    log2_enrich - log2(observed / expected) under independence
    pval      - hypergeometric upper tail, Bonferroni corrected over all cells
    """
    counts = create_confusion_matrix(data, rows=a, cols=b)

    n_a, n_b = counts.sum(1), counts.sum(0)
    N = int(counts.values.sum())
    inter = counts.values.astype(float)

    union = n_a.values[:, None] + n_b.values[None, :] - inter
    expected = np.outer(n_a.values, n_b.values) / N

    frame = lambda x: pd.DataFrame(x, index=counts.index, columns=counts.columns)
    with np.errstate(divide='ignore', invalid='ignore'):
        jaccard = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        log2_enrich = np.where(inter > 0, np.log2(np.maximum(inter, 1e-12) / expected), np.nan)

    pval = hypergeom.sf(inter - 1, N, n_a.values[:, None], n_b.values[None, :])
    pval = np.clip(pval * inter.size, 0, 1)  # Bonferroni over the whole table

    return {
        'a': a, 'b': b, 'N': N,
        'counts': counts,
        'jaccard': frame(jaccard),
        'fwd': counts.div(n_a, axis=0),
        'bwd': counts.div(n_b, axis=1),
        'log2_enrich': frame(log2_enrich),
        'pval': frame(pval),
    }


metrics = {(a, b): block_overlap_metrics(d, a, b) for a, b in chain_pairs}


def metrics_to_long(m):
    """One row per (block of `a`, block of `b`) pair, every combination kept.

    The matrices in `m` all share the same index/columns, so they can be
    flattened in row-major order onto a single MultiIndex without a join.
    Chain names become values rather than column names, which is what lets the
    three pairwise comparisons live in one dataframe.
    """
    counts, n_a, n_b = m['counts'], m['counts'].sum(1), m['counts'].sum(0)
    idx = pd.MultiIndex.from_product([counts.index, counts.columns],
                                     names=['block_a', 'block_b'])
    out = pd.DataFrame({k: m[k].to_numpy().ravel() for k in
                        ('counts', 'jaccard', 'fwd', 'bwd')}, index=idx).reset_index()
    out = out.rename(columns={'counts': 'n_int'})
    out.insert(0, 'a', m['a'])
    out.insert(1, 'b', m['b'])
    out.insert(4, 'n_a', out['block_a'].map(n_a).astype(int))
    out.insert(5, 'n_b', out['block_b'].map(n_b).astype(int))
    return out[['a', 'b', 'block_a', 'block_b', 'n_a', 'n_b',
                'n_int', 'jaccard', 'fwd', 'bwd']]


block_match = pd.concat([metrics_to_long(m) for m in metrics.values()],
                        ignore_index=True)

# remove zeor overlap rows 
block_match = block_match[block_match['n_int'] > 0]

# %% examine LR reliability across all chains 

pairs = pd.read_csv('init_data/pairs-2022-02-14.csv')
pairs['leftid'] = pairs['leftid'].astype(int)
pairs['rightid'] = pairs['rightid'].astype(int)
pairds = {lft:idx for idx, lft in enumerate(pairs['leftid'])} # all duplicates in left hemi
pairs['pairid'] = pairs['leftid'].map(pairds)
pairs['pairid'] = pairs['pairid'].astype(int)
skid_to_pair = dict(zip(pairs['leftid'], pairs['pairid']))
skid_to_pair.update(dict(zip(pairs['rightid'], pairs['pairid'])))

d['pairid'] = d['skeleton_id'].map(skid_to_pair)

from collections import defaultdict
matched_pairs = defaultdict(list)
unmatched_pairs = defaultdict(list)
for pairid, group in d.groupby('pairid'):
    if group.shape[0] != 2:
        continue
    for chain in chains:
        if group[chain].iloc[0] == group[chain].iloc[1]:
            matched_pairs[chain].append(pairid)
        else:
            unmatched_pairs[chain].append(pairid)

LR_match = pd.DataFrame(columns=['chain', 'matched_pairs', 'unmatched_pairs'])
LR_match['chain'] = list(matched_pairs.keys())
LR_match['matched_pairs'] = [matched_pairs[chain] for chain in LR_match['chain']]
LR_match['unmatched_pairs'] = [unmatched_pairs[chain] for chain in LR_match['chain']]
LR_match['n_matched'] = LR_match['matched_pairs'].apply(len)
LR_match['n_unmatched'] = LR_match['unmatched_pairs'].apply(len)

fig, ax = plt.subplots(1,1)
ax.bar(LR_match['chain'], LR_match['n_matched'], label='Matched', color='forestgreen')
ax.bar(LR_match['chain'], LR_match['n_unmatched'], bottom=LR_match['n_matched'], label='Unmatched', color='crimson')
ax.set_ylabel('Matched LR pairs')
ax.legend()

# %% examine sensories split 
import ast
sensory = pd.read_csv('init_data/sensory_neurons.csv')
sensory['skids'] = sensory['skids'].apply(ast.literal_eval)
skid_to_sense = dict()
for mod in sensory['modality'].unique():
    sensskids = sensory[sensory['modality'] == mod]['skids'].values[0]
    for skid in sensskids:
        skid_to_sense[skid] = mod

d['ct_with_sens'] = d.apply(lambda row: skid_to_sense.get(row['skeleton_id'], row['celltype']), axis=1)

fig, ax = plt.subplots(1, 3, figsize=(17, 7))
colormax = 230
for e, chain in enumerate(chains):
    cm = create_confusion_matrix(d, rows='ct_with_sens', cols=chain)
    # reorder cm so modalities are at the top 
    cm = cm.reindex(index=sorted(cm.index, key=lambda x: (x not in sensory['modality'].unique(), x)))

    # blank anything that rounds to 0.00 - row-normalised cells are mostly noise
    frac = cm.to_numpy()
    annot = np.where(frac < 1, "", np.char.mod('%.0f', frac))
    # fmt='' stops seaborn re-formatting the strings we just built
    sns.heatmap(cm, annot=annot, fmt='', cmap='Purples', ax=ax[e],
                vmin=None, vmax=colormax)
    ax[e].set_title(f'Confusion Matrix for {chain}')
    for _, spine in ax[e].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
for a in ax[1:]:
    a.set_yticks([])
    a.set_ylabel('')
for a in ax[:-1]:
    a.collections[0].colorbar.remove()
# %%
