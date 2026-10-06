''' To more carefully examine individual cells that are split into clusters 
different from their main celltype - across chains'''

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

mbnames = pd.read_csv('init_data/skid_to_name_wMB.csv')
mbnames['skid'] = mbnames['skid'].astype(int)
skid_to_name = dict(zip(mbnames['skid'], mbnames['name']))

skid_to_sens = {}
for mod, skids in zip(sens['modality'], sens['skids']):
    for skid in skids:
        skid_to_sens[skid] = mod

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

#%% 

''' Potential examples from looking at celltype block analysis:
-MBIN/MBONs
-MB-FBNs
-CNs
-gustatory clusters with or without visual neurons 
-mechano and noci 
'''

#%% label d futher with families 

chain25_map = {
    1: 'MB core',
    0: 'MB feedback/CN',
    5: 'Olfactory feedforward',
    6: 'Somatomotor',
    3: 'Premotor/descending',
    9: 'Premotor/descending',
    10: "Early sensory–enteric",
    8: "Early sensory–PN-linked",
}

chain39_map = {
    1: 'MB core',
    2: 'MB feedback/CN',
    3: 'Somatomotor',
    0: 'Olfactory feedforward',
    4: 'PN relay',
    6: 'LHN integrator',
    7: 'Premotor/descending' 
}

chain83_map = {
    0: 'MB core',
    1: 'MB feedback/CN',
    3: 'Olfactory feedforward',
    4: 'PN relay',
    7: 'LHN integrator',
    5: "Early sensory–enteric",
    9: "Early sensory–PN-linked",
    6: 'Premotor/descending' 
}

d['25clust'] = d['chain25'].map(chain25_map)
d['39clust'] = d['chain39'].map(chain39_map)
d['83clust'] = d['chain83'].map(chain83_map)

d['name'] = d['skeleton_id'].map(skid_to_name)
d['sens_1st'] = d['skeleton_id'].map(skid_to_sens)

#%% MBINs/MBONs

mbins = d[d['celltype']=='MBINs']
mbons = d[d['celltype']=='MBONs']

# %%

cns = d[d['celltype']=='CNs']
# %%

gust = d[d['sens_1st'].isin(['gustatory-external', 'gustatory-pharyngeal'])]
gust['matched'] = gust['25clust'] == gust['83clust']
# %%
