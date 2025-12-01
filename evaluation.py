import torch
from modules.geom_autoencoder import GeomVAE, EncoderwithPredictionHead
from modules.submodules import LatticeNormalizer
from torch_geometric.data import Data
from utils.lattice_utils import classify_nodes_with_geometry


def load_model(load_name, checkpoint_root, device):
    # ----------------------------- Load model
    normalizer = LatticeNormalizer()
    AEmodel = GeomVAE(
        normalizer=normalizer,
        max_node_num=30,
        latent_dim=128,
        edge_sample_threshold=0.5,
        is_variational=True,
        is_disent_variational=True,
        is_condition=True,
        condition_dim=12,
        disentangle_same_layer=True
    )

    # Load pretrained AE weights
    AEmodel.load_state_dict(torch.load(f"{checkpoint_root}/{load_name}/best_ae_model.pt", map_location=device))
    normalizer = AEmodel.normalizer
    normalizer = normalizer.to(device)

    model = EncoderwithPredictionHead(
        AEmodel=AEmodel,
        latent_dim=128,
        condition_dim=12,
    )
    model.load_state_dict(torch.load(f"{checkpoint_root}/{load_name}/best_predictor_model.pt", map_location=device))
    model = model.to(device)
    model.eval()

    return model, normalizer


def predict_single_graph(model, normalizer, data, device='cpu', denormalize=True):
    """
    Predict properties for a single lattice structure.

    Args:
        model: EncoderwithPredictionHead model (already .eval())
        normalizer: LatticeNormalizer (same as used in training)
        data: a torch_geometric.data.Data object
        device: 'cuda' or 'cpu'
        denormalize: whether to restore to real physical values

    Returns:
        y_pred: Tensor of shape [1, 12], young [0:3], shear [3:6], poisson [6:12]
    """
    model = model.to(device)
    normalizer = normalizer.to(device)
    model.eval()

    data = data.to(device)
    z = data.node_type
    coords = data.frac_coords
    edge_index = data.edge_index
    batch = torch.zeros(z.size(0), dtype=torch.long, device=device)
    lengths = data.lengths
    angles = data.angles
    num_atoms = data.num_atoms

    lengths_normed, angles_normed = normalizer(lengths.to(device), angles.to(device))

    with torch.no_grad():
        y_pred = model(
            z=z.to(device),
            coords=coords.to(device),
            edge_index=edge_index.to(device),
            batch=batch,
            lengths_normed=lengths_normed,
            angles_normed=angles_normed,
            num_atoms=num_atoms.to(device),
            denormalize=denormalize
        )

    return y_pred.cpu().squeeze(0)


def construct_input_data(frac_coords,  edge_index, lattice_angles, lattice_lengths):
    try:
        node_labels = classify_nodes_with_geometry(frac_coords, edge_index)
        z = torch.LongTensor(torch.argmax(node_labels, dim=-1) + 1)

    except:
        Warning("No geometry information, using default node labels")
        z = torch.ones(frac_coords.shape[0], dtype=torch.long)
    num_atoms = torch.LongTensor([frac_coords.shape[0]])
    return Data(frac_coords=frac_coords,node_type=z, edge_index=edge_index,
         num_atoms=num_atoms,lengths=lattice_lengths.unsqueeze(0), angles=lattice_angles.unsqueeze(0))








if __name__ == '__main__':
    # ------------------------------- Example  data
    frac_coords = torch.tensor([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [1.0, 1.0, 0.0],  # 2
        [0.0, 1.0, 0.0],  # 3
        [0.0, 0.0, 1.0],  # 4
        [1.0, 0.0, 1.0],  # 5
        [1.0, 1.0, 1.0],  # 6
        [0.0, 1.0, 1.0],  # 7
    ], dtype=torch.float)
    # edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom square
        # [4, 5], [5, 6], [6, 7], [7, 4],  # top square
        # [0, 4], [1, 5], [2, 6], [3, 7]  # vertical connections
    ]
    edge_index = torch.tensor(
        # [[i, j] for i, j in edges] + [[j, i] for i, j in edges],
        edges,
        dtype=torch.long
    ).t()
    lattice_lengths = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float)  # a, b, c
    lattice_angles = torch.tensor([90.0, 90.0, 90.0], dtype=torch.float)  # alpha, beta, gamma

    # ----------------------------- 1. Construct Data
    data = construct_input_data(frac_coords, edge_index, lattice_angles, lattice_lengths)
    # --------------------------- settings
    checkpoint_root = './checkpoints'
    load_name = '30_frac'  # checkpoint folder name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ------------------------------ 2. Load model
    model, normalizer = load_model(load_name, checkpoint_root, device)

    # ----------------------- 3. prediction
    y_pred = predict_single_graph(model, normalizer, data, device='cuda')
    print("Predicted properties:", y_pred)