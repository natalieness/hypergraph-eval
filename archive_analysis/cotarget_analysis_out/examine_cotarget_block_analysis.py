import numpy as np 
import pandas as pd
import ast 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl 

#%%
out_folder = 'cotarget_analysis_out'
d = pd.read_csv(f'{out_folder}/empirical_null_global_summary.csv')
print(d)

#%%

e = pd.read_csv(f'{out_folder}/all_block_pairs_vs_empirical_null.csv')
print(e)

#%%
f = pd.read_csv(f'{out_folder}/compression_summary.csv')

# %%

hb = pd.read_csv(f'data_mohammad/20260601_details_predicted_block.csv')
hb.drop(columns=["Unnamed: 0"], inplace=True)

# %%
# Stacked bar chart of cell type composition per block
mpl.rcParams.update({
    'font.size': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 18,
})
green_types  = ['sensories', 'ascendings']
blue_types   = ['dVNCs', 'dSEZs', 'RGNs']
red_types    = ['PNs', 'PNs-somato', 'LHNs', 'LNs', 'CNs', 'FFNs', 'KCs',  'MB-FBNs',
                'MBINs', 'MBONs',  'pre-dSEZs', 'pre-dVNCs']

green_colors = ["#11ac4f", '#78c893']
blue_colors  = ["#174bb3", '#4a7fd4', "#7da7e1"]
red_colors   = [
    "#d6c548", "#ebc60c", "#f1bb55", "#dc6e00",
    "#f6962fff", "#f0b070", "#8a2b62", "#e16faa",
    "#e22064", "#ce5280", "#3affeb", "#04c3e1",
]

celltypes_ordered = green_types + red_types + blue_types 
color_map = dict(zip(celltypes_ordered,
                     green_colors + red_colors + blue_colors))

hb_blocks = hb.dropna(subset=['predicted_block']).copy()
hb_blocks['predicted_block'] = hb_blocks['predicted_block'].astype(int)

counts = (hb_blocks
          .groupby(['predicted_block', 'celltype'])
          .size()
          .unstack(fill_value=0))
counts = counts.reindex(columns=celltypes_ordered, fill_value=0)

fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(len(counts))
for ct in celltypes_ordered:
    vals = counts[ct].values
    ax.bar(counts.index, vals, bottom=bottom,
           color=color_map[ct], label=ct, width=0.8)
    bottom += vals

ax.set_xlabel('Block')
ax.set_ylabel('Number of presynaptic neurons')
ax.set_title('Cell type composition per block')
ax.set_xticks(counts.index)
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig('figs/block_celltype_composition.svg', bbox_inches='tight')


#%% for ease of interpretation name the blocks based on cell types 

block_to_name = {
    0: 'sens/dSEZ/PN/LN',
    1: 'LHN/LN/pre-DN',
    2: 'dSEZ/PN/sens/pre-dSEZ',
    3: 'RGN/sens/DN',
    4: 'PN-somato/pre-dVNCs/asc/dVNCs',
    5: 'sens/dSEZ/PN',
    6: 'pre+dVNCs,PNsomato,MBFBNs',
    7: 'pre-dVNC/CN/FFN/MB-FBNs',
    8: 'sensories+minorRGN/FFN/asc',
    9: 'PNs+minorDN/LN',
    10: 'MBcore',
    11: 'sensories+minordSEZ',
    12: 'MB-FBN/LHN/CN/FFN',
    13: 'PN/sens/LN+LadderMBINs',
    14: 'pre+realDNs',
    15: 'PN/predVNC/CN/dSEZ',
    16: 'predVNC/DNs/asc',
}

# %%
g = pd.read_csv(f'{out_folder}/block_pair_enrichment_vs_per_source_abundance_null.csv')

col = 'log2_obs_exp'
g_heatmap = g.copy()
# turn into block_a = rows, block_b = columns, values = log2_obs_exp
g_heatmap = g_heatmap.pivot(index='block_a', columns='block_b', values=col)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(g_heatmap, annot=False, fmt='.2f', cmap='coolwarm', center=0, ax=ax, square=True)
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(2)
ax.set_xticklabels([block_to_name.get(i, str(i)) for i in g_heatmap.columns], rotation=45, ha='right')
ax.set_yticklabels([block_to_name.get(i, str(i)) for i in g_heatmap.index], rotation=0)
# %%
