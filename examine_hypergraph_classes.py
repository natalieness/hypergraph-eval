import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
#%% import some connectome data 
sensory_neurons = pd.read_csv("init_data/sensory_neurons.csv")
sensory_neurons['skids'] = sensory_neurons['skids'].apply(ast.literal_eval)

firstorder = sensory_neurons[['modality', 'skids']].explode('skids')
skid_to_modality = dict(zip(firstorder['skids'], firstorder['modality']))

con = pd.read_csv('init_data/connector_details2025.csv')
con['postsynaptic_to'] = con['postsynaptic_to'].apply(ast.literal_eval)
con['presynaptic_to'] = con['presynaptic_to'].astype(int)

#%%

data = pd.read_csv("data_mohammad/20260601_details_predicted_block.csv")
data.drop(columns=["Unnamed: 0"], inplace=True)

ct_names = data['celltype'].unique()
pred_nt = data['predicted_neurotransmitter'].unique()
data['predicted_block'] = pd.to_numeric(data['predicted_block'], errors='coerce').astype('Int64')
blocks = data['predicted_block'].unique()

def create_confusion_matrix(data, rows='celltype', cols='predicted_block'):
    confusion_matrix = pd.crosstab(data[rows], data[cols])
    return confusion_matrix

# Create confusion matrix
conf_matrix = create_confusion_matrix(data)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', ax=ax, square=True)
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(2)


# %% examine some subgrouping - sensory neurons

data_sens = data[data['celltype'] == 'sensories'].reset_index(drop=True)
data_sens['mod'] = data_sens['skeleton_id'].map(skid_to_modality)
sens_conf = create_confusion_matrix(data_sens, rows='mod', cols='predicted_block')
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(sens_conf, annot=True, fmt='d', cmap='Purples', ax=ax, square=True)
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(2)

# %% examine olfactory split across groups

data_olf = data_sens[data_sens['skeleton_id'].isin(sensory_neurons[sensory_neurons['modality'] == 'olfactory']['skids'].explode())].reset_index(drop=True)
data_olf.sort_values(by='predicted_block', inplace=True)
data_olf.reset_index(drop=True, inplace=True)
data_olf['base_name'] = data_olf['name'].apply(lambda x: x.replace(' left', '').replace(' right', ''))

# get number of base names split across blocks vs contained within a single block
base_name_counts = data_olf.groupby('base_name')['predicted_block'].nunique()
split_base_names = base_name_counts[base_name_counts > 1].index
contained_base_names = base_name_counts[base_name_counts == 1].index    
print(f"Number of base names split across blocks: {len(split_base_names)}")
print(f"Number of base names contained within a single block: {len(contained_base_names)}")



# %% examine different cell types - MB-FBNs

data_mbfbns = data[data['celltype'] == 'MB-FBNs']
data_mbfbns.sort_values(by='predicted_block', inplace=True)
data_mbfbns.reset_index(drop=True, inplace=True)
data_mbfbns['base_name'] = data_mbfbns['name'].apply(lambda x: x.replace('_left', '').replace('_right', '').replace('-left', '').replace('-right', '').replace(' left', '').replace(' right', ''))
base_name_counts_mbfbns = data_mbfbns.groupby('base_name')['predicted_block'].nunique()
split_base_names_mbfbns = base_name_counts_mbfbns[base_name_counts_mbfbns > 1].index
contained_base_names_mbfbns = base_name_counts_mbfbns[base_name_counts_mbfbns == 1].index    
print(f"Number of MB-FBN base names split across blocks: {len(split_base_names_mbfbns)}")
print(f"Number of MB-FBN base names contained within a single block: {len(contained_base_names_mbfbns)}")

fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=data_mbfbns, x='predicted_block', hue='predicted_neurotransmitter', ax=ax, palette='Set2')
ax.set_ylabel('Number of MB-FBN neurons')

# %%
con_from_mbfbns = con[con['presynaptic_to'].isin(data_mbfbns['skeleton_id'])]

# %%
con_from_mbfbns.to_csv('mbfbn_connectors.csv', index=False)
data_mbfbns.to_csv('mbfbn_neurons_hyperblocks.csv', index=False)

# %%
