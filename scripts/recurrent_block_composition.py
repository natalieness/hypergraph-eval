import re, ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle, Patch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
#%% data
d = pd.read_csv('data_mohammad/202609_fullmodel/details_with_labels.csv', index_col=0)
chains = ['chain25', 'chain39', 'chain83']
sens = pd.read_csv('init_data/sensory_neurons.csv')
sens['skids'] = sens['skids'].apply(ast.literal_eval)
con = pd.read_csv('init_data/connector_details2025.csv', usecols=['presynaptic_to', 'postsynaptic_to'])
con['postsynaptic_to'] = con['postsynaptic_to'].apply(ast.literal_eval)

#%% recurring family <-> cluster correspondence across chains
families = pd.DataFrame([
    ('MB core', 'c1', 'c1', 'c0', 'stable'),
    ('MB feedback/CN', 'c0', 'c2', 'c1', 'stable'),
    ('Olfactory feedforward', 'c5', 'c0', 'c3', 'stable'),
    ('PN relay', 'c4 part', 'c4', 'c4', 'split'),
    ('LHN integrator', 'c2/c4/c7', 'c6', 'c7', 'split'),
    ('Somatomotor', 'c6', 'c3', 'c2 part', 'merged'),
    ('Premotor/descending', 'c3 + c9', 'c7', 'c6', 'split'),
    ('Early sensory–enteric', 'c10', 'c5 part', 'c5', 'merged'),
    ('Early sensory–PN-linked', 'c8', 'c5 part', 'c9', 'merged'),
], columns=['family', *chains, 'status']).set_index('family')
in_fam = lambda f, c: d[c].isin([int(x) for x in re.findall(r'\d+', families.at[f, c])])

# IOU of the family's neurons between each pair of chains
iou_cols = [f'{a[5:]}:{b[5:]}' for a, b in zip(chains, chains[1:] + chains[:1])]
for k, (a, b) in zip(iou_cols, zip(chains, chains[1:] + chains[:1])):
    families[k] = [(in_fam(f, a) & in_fam(f, b)).sum() / (in_fam(f, a) | in_fam(f, b)).sum() for f in families.index]

# neuron -> family if its clusters point to that family in >=2 chains (unique max)
votes = pd.DataFrame({f: sum(in_fam(f, c) for c in chains) for f in families.index})
top = votes.max(1)
d['family'] = votes.idxmax(1).where((top >= 2) & (votes.eq(top, axis=0).sum(1) == 1))
print(d['family'].value_counts(dropna=False))

#%% plot A: correspondence table
mpl.rcParams.update({
    'font.size': 12,
    'legend.fontsize': 12
})
stat_col = {'stable match': '#dbeafe', 'split across clusters': '#dcf2e3', 'merged with another family': '#fde4d6'}
fam_col = dict(zip(['stable', 'split', 'merged'], stat_col.values()))
fig, ax = plt.subplots(figsize=(9.5, 0.52 * len(families) + 1.2))
kw, hkw, ix = dict(va='center',  color='#333'), dict(va='center', color='#999'), [7.6, 8.4, 9.2]
for i, (f, r) in enumerate(families.iterrows()):
    ax.text(0, i, f, **kw)
    for j, c in enumerate(chains):
        ax.add_patch(Rectangle((2.4 + 1.6 * j, i - .3), 1.4, .6, color=fam_col[r.status], lw=0))
        ax.text(3.1 + 1.6 * j, i, r[c], ha='center', **kw)
    for x, k in zip(ix, iou_cols): ax.text(x, i, f'{r[k]:.2f}', ha='center', **kw)
    ax.axhline(i + .5, color='#eee', lw=0.5)
for x, h in zip([0, 3.1, 4.7, 6.3], ['Family', *chains]): ax.text(x, -1.1, h, ha='left' if x == 0 else 'center', **hkw)
ax.text(np.mean(ix), -1.5, 'IOU', ha='center', **hkw); ax.plot([ix[0] - .35, ix[-1] + .35], [-1.25] * 2, color='#ddd', lw=0.5)
for x, k in zip(ix, iou_cols): ax.text(x, -0.85, k, ha='center', **hkw)
ax.axhline(-.5, color='#ddd', lw=0.5)
ax.set(xlim=(0, 9.8), ylim=(len(families) - .3, -1.9)); ax.axis('off')
ax.legend(handles=[Patch(color=v, label=k) for k, v in stat_col.items()], loc='upper left', bbox_to_anchor=(0, 0), ncol=3, frameon=False)
ax.set_title('Recurring cluster correspondence', loc='center')
fig.tight_layout()

#%% sensory order: 1 + hops from nearest sensory neuron, per modality via virtual source nodes
SYN = 3  # min synapses for an edge
e = con.explode('postsynaptic_to').dropna().astype(int).value_counts().reset_index()
e = e[(e['count'] >= SYN) & (e.presynaptic_to != e.postsynaptic_to)]
sx = sens.explode('skids').astype({'skids': int})
nodes = np.unique(np.r_[e.presynaptic_to, e.postsynaptic_to, sx.skids, d.skeleton_id])
n, mods = len(nodes), list(sens.modality)
rows = np.r_[np.searchsorted(nodes, e.presynaptic_to), n + sx.modality.map({m: k for k, m in enumerate(mods)})]
cols = np.r_[np.searchsorted(nodes, e.postsynaptic_to), np.searchsorted(nodes, sx.skids)]
A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n + len(mods),) * 2)
dist = shortest_path(A, directed=True, unweighted=True, indices=np.arange(n, n + len(mods)))[:, :n]
dd = pd.DataFrame(dist.T, index=nodes, columns=mods).loc[d.skeleton_id].set_index(d.index)
d['order'] = dd.min(1).where(d.celltype != 'sensories', 1)
lead = dd.eq(dd.min(1), axis=0) & np.isfinite(dd)  # modalities reaching neuron at its order

#%% plot B: family composition, sensory order, leading modalities
rank_col, grey = ['#4a9eed', '#f28e3c', '#6cc774', '#ec7fb4'], '#d6d6d6'
fig, axs = plt.subplots(len(families), 3, figsize=(10, 0.95 * len(families) + 0.6), gridspec_kw={'width_ratios': [3, 1.3, 1.1]})

def hbar(ax, vals, colors, inside=False):
    x = 0
    for v, c in zip(vals, colors):
        ax.barh(0.65, v, left=x, height=0.45, color=c)
        if inside and v >= 0.1: ax.text(x + v / 2, 0.65, f'{v:.0%}', ha='center', va='center')
        x += v
    ax.set(xlim=(0, 1), ylim=(-0.4, 1)); ax.axis('off')

for i, f in enumerate(families.index):
    g = d[d.family == f]
    ct = g.celltype.value_counts(normalize=True)
    parts = [*ct.iloc[:4].items(), ('other', ct.iloc[4:].sum())]
    hbar(axs[i, 0], [p for _, p in parts], rank_col + [grey])
    t = axs[i, 0].text(0, 0.05, '', fontsize=8, va='center')
    for (k, p), c in zip(parts, rank_col + ['grey']):
        t = axs[i, 0].annotate(f'{k} {p:.0%}  ', xy=(1, 0), xycoords=t, va='bottom', color=c)
    axs[i, 0].text(-0.02, 0.5, f'{f}\n(n={len(g)})', ha='right', va='center')
    o = g.order.value_counts(normalize=True).reindex([1, 2, 3, 4], fill_value=0)
    hbar(axs[i, 1], [*o, 1 - o.sum()], rank_col + [grey], inside=True)
    m = lead.loc[g.index].mean().nlargest(2)
    axs[i, 2].text(0, 0.5, ''.join(f'{k} {v:.0%} \n' for k, v in m.items()), va='center'); axs[i, 2].axis('off')
for a, h in zip(axs[0], ['Cell-type composition', 'Sensory order', 'Leading modalities']): a.set_title(h, loc='left',  color='grey')
fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(rank_col, ['1st', '2nd', '3rd', '4th'])], loc='lower left', ncol=4, frameon=False, title='Sensory order', alignment='left')
fig.subplots_adjust(left=0.16, right=0.98, top=0.95, bottom=0.06, hspace=0.3, wspace=0.08)
plt.show()

#%% connectivity partners per family: every postsynaptic_to entry is a target occurrence (repeats kept)
ct_all = pd.read_csv('init_data/neuron_details_with_nt.csv').set_index('skeleton_id').celltype
nm = lambda c: {'sensories': 'sensory', 'PNs-somato': 'PN-somato', 'MB-FBNs': 'FBN'}.get(c, c.rstrip('s'))
skid_fam = d.set_index('skeleton_id').family
ex = con.reset_index(names='cid').explode('postsynaptic_to').dropna().astype(int)
ex['pos'] = ex.groupby('cid').cumcount()
ex['pre_ct'], ex['post_ct'] = ex.presynaptic_to.map(ct_all).map(nm, na_action='ignore'), ex.postsynaptic_to.map(ct_all).map(nm, na_action='ignore')
ex['pre_fam'], ex['post_fam'] = ex.presynaptic_to.map(skid_fam), ex.postsynaptic_to.map(skid_fam)
pp = ex.merge(ex[['cid', 'pos', 'post_ct']], on='cid', suffixes=('', '_peer')).query('pos != pos_peer').dropna(subset=['post_ct', 'post_ct_peer'])

THRESH = 0.1  # show every category making up > THRESH of that family's column
def top(s): v = s.dropna().value_counts(normalize=True); return [f'{k} {p:.0%}' for k, p in v[v > THRESH].items()]
part = pd.DataFrame({f: {
    'Consists of': top(d[d.family == f].celltype.map(nm)),
    'Targeted by': top(ex[ex.post_fam == f].pre_ct),
    'Targets': top(ex[ex.pre_fam == f].post_ct),
    'Outgoing co-target pairs': top(pp[(pp.pre_fam == f) & (pp.pos < pp.pos_peer)][['post_ct', 'post_ct_peer']].apply(lambda r: ' + '.join(sorted(r)), axis=1)),
    'Incoming source → peer': top(pp[pp.post_fam == f].dropna(subset='pre_ct').pipe(lambda x: x.pre_ct + ' → ' + x.post_ct_peer)),
} for f in families.index}).T

#%% plot C: partner table
chip_col = dict(zip(part.columns, ['#eeeeee', '#dbeafe', '#fde4d6', '#dcf2e3', '#fbe0ea']))
L = 0.45  # line spacing for stacked chips
nrow = part.map(len).max(1).clip(lower=1)
y0 = np.r_[0, np.cumsum(L * nrow + 0.4)]  # row tops, height grows with stacked chips
fig, ax = plt.subplots(figsize=(14, 0.6 * y0[-1] + 1))
xs = [0, 0.17, 0.31, 0.45, 0.59, 0.79]
for i, (f, r) in enumerate(part.iterrows()):
    ax.text(0, y0[i], f, va='bottom')
    for x, c in zip(xs[1:], part.columns):
        for j, item in enumerate(r[c]):
            ax.text(x, y0[i] + L * j, item, va='bottom', bbox=dict(boxstyle='round,pad=0.3', fc=chip_col[c], ec='none'))
    ax.axhline(y0[i + 1] - 0.6, color='#eee', lw=0.8)
for x, h in zip(xs, ['Family', *part.columns]): ax.text(x, -0.7, h, color='grey',  va='bottom')
ax.set(xlim=(0, 1), ylim=(y0[-1], -0.9)); ax.axis('off')
# add threshold as text at the bottom 
ax.text(0, y0[-1] + 0.2, f'Threshold for inclusion: {THRESH:.0%}', color='grey', va='bottom', fontsize=12)
fig.tight_layout()
plt.show()
# %%
