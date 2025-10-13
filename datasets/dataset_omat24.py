import torch
import torch.nn.functional as F
from tqdm import tqdm
# from datasets import LatticeModulus
from modules.geom_autoencoder import GeomEncoder, GeomDecoder, GeomVAE
from torch_geometric.loader import DataLoader
from modules.submodules import LatticeNormalizer
from utils.ldm_utils import unpadding_max_num_node2graph
from visualization.vis import visualizeLattice
import numpy as np
import datetime
import wandb
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import radius_graph
import os
import ase.neighborlist

class OMAT24Dataset(Dataset):
    def __init__(self, ase_dataset, cutoff=5.0, transform=None, pre_transform=None, pre_filter=None, condition_dim=12):
        self.ase_dataset = ase_dataset
        self.cutoff = cutoff
        self.condition_dim = condition_dim
        self.lengths = []
        self.angles = []
        super().__init__(None, transform, pre_transform, pre_filter)
        
    def len(self):
        return len(self.ase_dataset)
    
    def get(self, idx):
        atoms = self.ase_dataset.get_atoms(idx)
        return self.convert_atoms_to_data(atoms, idx)
    

    def convert_atoms_to_data(self, atoms, idx=0):
        # Get atomic numbers and positions
        atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long)
        positions = torch.tensor(atoms.positions, dtype=torch.float)
        
        # Get cell information and convert to lengths and angles
        cell = atoms.get_cell()
        cell_tensor = torch.tensor(cell, dtype=torch.float)
        
        # Calculate lengths (a, b, c)
        lengths = torch.tensor([
            np.linalg.norm(cell[0]),
            np.linalg.norm(cell[1]),
            np.linalg.norm(cell[2])
        ], dtype=torch.float).view(1, 3)
        
        # Calculate angles (alpha, beta, gamma) in degrees
        angles = torch.tensor([
            np.degrees(np.arccos(np.dot(cell[1], cell[2]) / (np.linalg.norm(cell[1]) * np.linalg.norm(cell[2])))),
            np.degrees(np.arccos(np.dot(cell[0], cell[2]) / (np.linalg.norm(cell[0]) * np.linalg.norm(cell[2])))),
            np.degrees(np.arccos(np.dot(cell[0], cell[1]) / (np.linalg.norm(cell[0]) * np.linalg.norm(cell[1]))))
        ], dtype=torch.float).view(1, 3)
        
        # Store for normalizer
        self.lengths.append(lengths)
        self.angles.append(angles)
        # self.lengths = torch.cat(self.lengths, dim=0)
        # self.angles = torch.cat(self.angles, dim=0)

        # Get fractional coordinates
        cell_inv = np.linalg.inv(cell)
        frac_coords = torch.tensor(np.dot(positions, cell_inv), dtype=torch.float)
        
        # Create edge indices using radius graph
        edge_index = radius_graph(positions, r=self.cutoff)
        
        # Calculate node features based on atomic numbers
        # For simplicity, assume a max of 118 elements (periodic table)
        node_feat = torch.zeros((len(atomic_numbers), 4), dtype=torch.float)

        # Use atomic numbers directly for node types
        node_type = atomic_numbers

        # Create simple feature encoding based on atomic numbers
        # Feature 0: Normalized atomic number (1-118 -> 0-1)
        node_feat[:, 0] = atomic_numbers.float() / 118.0

        # Feature 1: Period in periodic table (approximate)
        # 1-2: Period 1, 3-10: Period 2, 11-18: Period 3, etc.
        periods = torch.floor(torch.sqrt(atomic_numbers.float())).clamp(1, 7)
        node_feat[:, 1] = periods / 7.0  # Normalize to [0,1]

        # Feature 2: Group (metals vs. non-metals, simplified)
        # Simple heuristic: higher atomic numbers tend to be metals
        node_feat[:, 2] = (atomic_numbers > 20).float()

        # Feature 3: Size/weight indicator
        node_feat[:, 3] = torch.log(atomic_numbers.float() + 1) / torch.log(torch.tensor(119.0))
        
        # Edge features (default to constant value)
        edge_feat = torch.ones((edge_index.size(1), 1), dtype=torch.float)
        
        # Get properties if available, otherwise create dummy values
        # For OMAT24, extract properties from the ASE atoms object if they exist
        # For example, trying to get properties from atoms.info
        
        stresses = atoms.get_stress()
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        # properties = torch.tensor([energy] + stresses.tolist(), dtype=torch.float).view(1, -1)

        # Construct PyG Data object
        data = Data(
            frac_coords=frac_coords,
            cart_coords=positions,
            node_feat=node_feat,
            node_type=node_type,
            edge_feat=edge_feat,
            edge_index=edge_index,
            num_nodes=len(atomic_numbers),
            num_atoms=len(atomic_numbers),
            num_edges=edge_index.size(1),
            lengths=lengths,
            angles=angles,
            vector=cell_tensor.view(1, -1),
            y=torch.FloatTensor([energy]).view(1, -1),
            energy=energy,
            stresses=torch.from_numpy(stresses).float(),
            forces=torch.from_numpy(forces).float(),
            to_jimages=None
        )

        return data
    
    def get_idx_split(self, dataset_size, train_size, valid_size, seed=42):
        """
        Split the dataset into training, validation, and test sets.
        """
        np.random.seed(seed)
        
        # Generate indices for each set
        indices = np.random.permutation(dataset_size)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size + valid_size]
        test_indices = indices[train_size + valid_size:]
        
        # Return as dictionary
        return {
            'train': train_indices,
            'valid': valid_indices,
            'test': test_indices
        }
    
    def collate(self, data_list):
        """Collate function similar to what LatticeTruss might use"""
        from torch_geometric.data import Batch
        return Batch.from_data_list(data_list)




