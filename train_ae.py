import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import LatticeModulus, OMAT24Dataset
from modules.geom_autoencoder import GeomEncoder, GeomDecoder, GeomVAE
from torch_geometric.loader import DataLoader
from modules.submodules import LatticeNormalizer
from utils.ldm_utils import unpadding_max_num_node2graph
from visualization.vis import visualizeLattice
import numpy as np
import datetime
import wandb

import os

MAX_NODE_NUM = 500
sample_max_num_nodes = 30
train_size = 100000
valid_size = 1
batch_size=256
condition_dim=1
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch = 5000
# device='cpu'
is_condition=True
is_disent_variational= False
disentagle_same_layer=True
# load_name = 'vae_cond_128_beta001_dis_same_100_frac'
# save_name = 'vae_cond_128_beta001_dis_same_100_frac_'
load_name='omat24_rattle2'
save_name = 'omat24_rattle2'
root = '/home/grads/jianpengc/projects/materials/R_MetaSymbO/'
use_wandb=True
latent_dim=128

# data_root = '/home/jianpengc/datasets/metamaterial/'
data_root = '/home/grads/jianpengc/datasets/omat24/rattled-relax'

# --------------
#   Load data
# --------------
print('Loading Data ...')
# dataset = LatticeModulus(f'{data_root}/LatticeModulus', file_name='data')
# # dataset = LatticeModulus('D:\\项目\\Material design\\code_data\\data\\LatticeModulus',file_name='data_new')
# indices = []
# for i, data in enumerate(tqdm(dataset)):
#     if data.num_atoms <= sample_max_num_nodes and data.num_edges <= sample_max_num_nodes * 2:
#         indices.append(i)
# dataset = dataset[indices]
# print('All data size', len(dataset))
# split_idx = dataset.get_idx_split(len(dataset), train_size=train_size, valid_size=valid_size, seed=42)
# print(split_idx.keys())
# print(dataset[split_idx['train']])
# train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[
#     split_idx['test']
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
from fairchem.core.datasets import AseDBDataset

dataset_path = data_root
config_kwargs = {} # see tutorial on additional configuration

ase_dataset = AseDBDataset(config=dict(src=dataset_path, **config_kwargs))

# Convert to PyG format
dataset = OMAT24Dataset(ase_dataset, cutoff=5.0, condition_dim=condition_dim)
print('All data size', len(dataset))
# indices = []
# for i in range(len(dataset)):
#     if dataset[i].num_atoms <= MAX_NODE_NUM:
#         indices.append(i)
# dataset = dataset[indices]
# print('Filtered data size', len(dataset))



# Split dataset
split_idx = dataset.get_idx_split(len(dataset), train_size=train_size, valid_size=valid_size, seed=42)
train_dataset = dataset[split_idx['train']]
valid_dataset = dataset[split_idx['valid']]
test_dataset = dataset[split_idx['test']]


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate)
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)

lengths = [i.lengths for i in train_dataset]
angles = [i.angles for i in train_dataset]
# max_num_nodes = max([data.num_atoms for data in train_dataset])
# print('Max num nodes:', max_num_nodes)

# --------------
#   Init model
# --------------
print('Init Model ...')


normalizer = LatticeNormalizer(lengths, angles)


model = GeomVAE(normalizer, max_node_num=MAX_NODE_NUM, latent_dim=latent_dim, edge_sample_threshold=0.5, is_variational=True,
                 is_disent_variational=is_disent_variational, is_condition=is_condition, condition_dim=condition_dim, disentangle_same_layer=disentagle_same_layer)
model = model.to(device)
normalizer = normalizer.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=.8)

print("Start training...")
if os.path.exists(root+f'/checkpoints/{load_name}/best_ae_model.pt'):
    model.load_state_dict(torch.load(root+f'/checkpoints/{load_name}/best_ae_model.pt', map_location=device))
if use_wandb:
    wandb.init(
            entity='jianpengc',
            project='OMAT24',
            name=save_name+'-'+datetime.datetime.now().strftime('%Y-%m-%d--%H:%M'),
        )

model.train()
model.train_model(train_loader, optimizer, device=device, num_epochs=epoch, beta=1, beta_geo=0.0001, scheduler=scheduler,
                  checkpoint_dir=root+f'/checkpoints/{save_name}/', save_every=100, use_wandb=use_wandb)


model.eval()

node_num_logit, node_num_pred, lengths_pred, angles_pred, latent = model.sample_lattice(num_samples=3, device=device)

lengths_pred, angles_pred = normalizer.denormalize(lengths_pred, angles_pred)

print("Sampled node numbers: ", node_num_pred)
print("Sampled lengths: ", lengths_pred)
print("Sampled angles: ", angles_pred)




