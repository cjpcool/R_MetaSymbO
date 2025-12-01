import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import LatticeModulus
from modules.geom_autoencoder import GeomEncoder, GeomDecoder, GeomVAE, EncoderwithPredictionHead
from torch_geometric.loader import DataLoader
from modules.submodules import LatticeNormalizer
from utils.ldm_utils import unpadding_max_num_node2graph
from visualization.vis import visualizeLattice
import numpy as np
import datetime
import wandb

import os

MAX_NODE_NUM = 100
sample_max_num_nodes = 100
train_size = 8000
valid_size = 40
batch_size=512
condition_dim=12
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch = 10000
# device='cpu'
is_condition=True
is_disent_variational=True
disentagle_same_layer=True
load_name = 'vae_cond_128_beta001_dis_same_100_frac'
save_name = 'vae_cond_128_beta001_dis_same_100_frac_young'
# load_name = '30_frac_new1'
# save_name = '30_frac_new1'
root = '.'
use_wandb=False
latent_dim=128

data_root = '/home/jianpengc/datasets/metamaterial/'





def calculate_metrics(pred, target):
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    """
    Calculate and return R^2, NRMSE, MAE for given prediction and target values.

    Parameters:
    pred (array-like): The predicted values.
    target (array-like): The true target values.

    Returns:
    tuple: A tuple containing R^2, NRMSE, and MAE.
    """

    # Calculate R^2 (coefficient of determination)
    r2 = r2_score(target, pred)

    # Calculate RMSE (root mean squared error) from MSE
    rmse = np.sqrt(mean_squared_error(target, pred))


    # Calculate NRMSE (normalized root mean squared error)
    # Normalize RMSE by dividing it by the range of the target values (max - min)
    range_target = np.max(target) - np.min(target)
    if range_target == 0:
        raise ValueError("The range of target values cannot be zero")
    nrmse = rmse / range_target

    # Calculate MAE (mean absolute error)
    mae = mean_absolute_error(target, pred)

    return r2, nrmse, mae





# --------------
#   Load data
# --------------
dataset = LatticeModulus(f'{data_root}/LatticeModulus', file_name='data')
indices = []
for i, data in enumerate(tqdm(dataset)):
    if data.num_atoms <= sample_max_num_nodes and data.num_edges <= sample_max_num_nodes * 2:
        indices.append(i)
dataset = dataset[indices]
print('All data size', len(dataset))
split_idx = dataset.get_idx_split(len(dataset), train_size=train_size, valid_size=valid_size, seed=42)
print(split_idx.keys())
print(dataset[split_idx['train']])
# print(dataset.y.max(dim=0), dataset.y.min(dim=0))
train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[
    split_idx['test']]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(train_dataset, batch_size=500, shuffle=False)



# --------------
#   Init model
# --------------
normalizer = LatticeNormalizer()


AEmodel = GeomVAE(normalizer, max_node_num=MAX_NODE_NUM, latent_dim=latent_dim, edge_sample_threshold=0.5, is_variational=True,
                 is_disent_variational=is_disent_variational, is_condition=is_condition, condition_dim=condition_dim, disentangle_same_layer=disentagle_same_layer)


print("Start training...")
AEmodel.load_state_dict(torch.load(root+f'/checkpoints/{load_name}/best_ae_model.pt', map_location=device))
normalizer = AEmodel.normalizer

y_mean = train_dataset.y.mean(dim=0).to(device).view(1, -1)
y_std = train_dataset.y.std(dim=0).to(device).view(1, -1)

# y_mean = y_mean[:, :3]
# y_std = y_std[:, :3]
model = EncoderwithPredictionHead(AEmodel, latent_dim=latent_dim, condition_dim=12, y_mean=y_mean, y_std=y_std)
if os.path.exists(root+f'/checkpoints/{load_name}/best_predictor_model.pt'):
    model.load_state_dict(torch.load(root+f'/checkpoints/{load_name}/best_predictor_model.pt', map_location=device))
model = model.to(device)
normalizer = normalizer.to(device)





model.eval()

target_young =[]
target_shear = []
target_poisson = []
pred_young = []
pred_shear = []
pred_poisson = []
with torch.no_grad():
    for batch_data in tqdm(test_loader):
        z, coords, edge_index, batch, lengths, angles, num_atoms = batch_data.node_type, batch_data.frac_coords, batch_data.edge_index, batch_data.batch, batch_data.lengths, batch_data.angles, batch_data.num_atoms
        z = z.to(device)
        coords = coords.to(device)
        edge_index = edge_index.to(device)
        lengths = lengths.to(device)
        angles = angles.to(device)
        num_atoms = num_atoms.to(device)
        batch = batch.to(device)
        y = batch_data.y.to(device)
        batch_size = y.shape[0]

        lengths_normed, angles_normed = normalizer(lengths, angles)

        y_pred = model(z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms, denormalize=True)
        young, shear, poisson = y_pred[:, :3], y_pred[:,3:6], y_pred[:, 6:]
        t_y, t_s, t_p = y[:, :3], y[:, 3:6], y[:, 6:]
        target_young.append(t_y)
        target_shear.append(t_s)
        target_poisson.append(t_p)

        pred_young.append(young)
        pred_shear.append(shear)
        pred_poisson.append(poisson)

target_young = torch.cat(target_young, dim=0).cpu().numpy()
target_shear = torch.cat(target_shear, dim=0).cpu().numpy()
target_poisson = torch.cat(target_poisson, dim=0).cpu().numpy()
pred_young = torch.cat(pred_young, dim=0).cpu().numpy()
pred_shear = torch.cat(pred_shear, dim=0).cpu().numpy()
pred_poisson = torch.cat(pred_poisson, dim=0).cpu().numpy()
r2, nrmse, mae =  calculate_metrics(pred_young, target_young)
print('young', r2, nrmse, mae,)

print('shear', calculate_metrics(pred_shear, target_shear))

print('poisson', calculate_metrics(pred_poisson, target_poisson))

exit()












if use_wandb:
    wandb.init(
            entity='jianpengc',
            project='GeomVAE_Predictor',
            name=save_name+'-'+datetime.datetime.now().strftime('%Y-%m-%d--%H:%M'),
        )





def evaluate():
    model.eval()

    target_young =[]
    target_shear = []
    target_poisson = []
    pred_young = []
    pred_shear = []
    pred_poisson = []
    with torch.no_grad():
        for batch_data in tqdm(test_loader):
            z, coords, edge_index, batch, lengths, angles, num_atoms = batch_data.node_type, batch_data.frac_coords, batch_data.edge_index, batch_data.batch, batch_data.lengths, batch_data.angles, batch_data.num_atoms
            z = z.to(device)
            coords = coords.to(device)
            edge_index = edge_index.to(device)
            lengths = lengths.to(device)
            angles = angles.to(device)
            num_atoms = num_atoms.to(device)
            batch = batch.to(device)
            y = batch_data.y.to(device)
            batch_size = y.shape[0]

            lengths_normed, angles_normed = normalizer(lengths, angles)

            y_pred = model(z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms, denormalize=True)
            young, shear, poisson = y_pred[:, :3], y_pred[:,3:6], y_pred[:, 6:]
            t_y, t_s, t_p = y[:, :3], y[:, 3:6], y[:, 6:]
            target_young.append(t_y)
            target_shear.append(t_s)
            target_poisson.append(t_p)

            pred_young.append(y_pred)
            pred_shear.append(shear)
            pred_poisson.append(poisson)

    target_young = torch.cat(target_young, dim=0).cpu().numpy()
    target_shear = torch.cat(target_shear, dim=0).cpu().numpy()
    target_poisson = torch.cat(target_poisson, dim=0).cpu().numpy()
    pred_young = torch.cat(pred_young, dim=0).cpu().numpy()
    pred_shear = torch.cat(pred_shear, dim=0).cpu().numpy()
    pred_poisson = torch.cat(pred_poisson, dim=0).cpu().numpy()
    r2, nrmse, mae =  calculate_metrics(pred_young, target_young)
    print('young', r2, nrmse, mae,)

    print('shear', calculate_metrics(pred_shear, target_shear))

    print('poisson', calculate_metrics(pred_poisson, target_poisson))




os.makedirs(root+f'/checkpoints/{save_name}', exist_ok=True)
model.train()
optimizer = torch.optim.Adam(model.Predictor.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=.8)
min_lr = 1e-8

best_loss = np.inf
for epoch in range(epoch):
    model.train()
    train_loss = 0.0
    for batch_data in tqdm(train_loader):

        z, coords, edge_index, batch, lengths, angles, num_atoms = batch_data.node_type, batch_data.frac_coords, batch_data.edge_index, batch_data.batch, batch_data.lengths, batch_data.angles, batch_data.num_atoms
        z = z.to(device)
        coords = coords.to(device)
        edge_index = edge_index.to(device)
        lengths = lengths.to(device)
        angles = angles.to(device)
        num_atoms = num_atoms.to(device)
        batch = batch.to(device)
        y = batch_data.y.to(device)
        batch_size = y.shape[0]

        lengths_normed, angles_normed = normalizer(lengths, angles)

        optimizer.zero_grad()
        y_pred = model(z, coords, edge_index, batch, lengths_normed, angles_normed, num_atoms, denormalize=False)
        # normalize y
        # print('before ', y_pred)
        # y_pred_6 = y_pred[:, :6]
        # y_pred_post = y_pred[:, 6:]
        # y_pred_6 = torch.abs(y_pred_6)
        # y_pred = torch.cat([y_pred_6, y_pred_post], dim=1)
        y = y[:, :3]
        # print('after', y_pred)
        y = (y - y_mean) / y_std
        loss = F.mse_loss(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_size

    train_loss /= len(train_loader.dataset)
    if scheduler.get_last_lr()[0] > min_lr:
        scheduler.step()

    # save the model
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), root+f'/checkpoints/{save_name}/best_predictor_model.pt')
        print(f"Best model saved at epoch {epoch} with loss {best_loss:.4f}")

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
        r2, nrmse, mae = evaluate()
        if use_wandb:
            wandb.log({'mae': mae})
    if use_wandb:
        wandb.log({"train_loss": train_loss})
        wandb.log({"epoch": epoch})
        wandb.log({"lr": scheduler.get_last_lr()[0]})
        





exit()