import re
import torch
from utils.lattice_utils import classify_nodes_with_geometry
import numpy as np


def parse_graph(text: str):
    text = text.strip()
    text = re.sub(r"^\s*```[a-zA-Z0-9]*\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    text = re.sub(r"^\s*~~~[a-zA-Z0-9]*\s*", "", text)
    text = re.sub(r"\s*~~~\s*$", "", text)

    lines = [ln.strip() for ln in text.splitlines()]

    # states
    IN_NONE, IN_Z, IN_COORD, IN_EDGE = 0, 1, 2, 3
    state = IN_NONE

    N_declared = None
    Z_list = []
    coords_list = []
    edges_list = []
    lengths = None
    angles = None

    # regex helpers
    float_pat = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
    int_pat   = re.compile(r"[-+]?\d+")
    bracket_nums = re.compile(r"\[([^\]]+)\]")

    def parse_three_floats_from_tuple(s: str):
        nums = float_pat.findall(s)
        if len(nums) >= 3:
            return [float(nums[0]), float(nums[1]), float(nums[2])]
        return None

    for ln in lines:
        low = ln.lower()

        if not ln:
            continue

        # section headers
        if low.startswith("node number"):
            # e.g., "Node number: 4"
            m = int_pat.search(ln)
            if m:
                N_declared = int(m.group(0))
            state = IN_NONE
            continue

        if "element z" in low:
            state = IN_Z
            continue

        if ("node fractional coordinates" in low) or ("node coordinates" in low):
            state = IN_COORD
            continue

        if low.startswith("edges"):
            state = IN_EDGE
            continue

        if low.startswith("lattice lengths"):
            # e.g., "Lattice lengths: [1, 1, 1]"
            m = bracket_nums.search(ln)
            if m:
                nums = [float(x) for x in float_pat.findall(m.group(1))]
                if len(nums) >= 3:
                    lengths = nums[:3]
            continue

        if low.startswith("lattice angles"):
            m = bracket_nums.search(ln)
            if m:
                nums = [float(x) for x in float_pat.findall(m.group(1))]
                if len(nums) >= 3:
                    angles = nums[:3]
            continue

        # content lines by state
        if state == IN_Z:
            # tolerate lines like "26" or "Z=26"
            m = int_pat.search(ln)
            if m:
                Z_list.append(int(m.group(0)))
            continue

        if state == IN_COORD:
            vals = parse_three_floats_from_tuple(ln)
            if vals is not None:
                coords_list.append(vals)
            continue

        if state == IN_EDGE:
            # tolerate "(i, j)" or "i j" etc.
            ints = [int(x) for x in int_pat.findall(ln)]
            if len(ints) >= 2:
                edges_list.append([ints[0], ints[1]])
            continue

        # otherwise ignore stray lines

    # --- finalize with defaults / sanity ---
    N = None
    if coords_list:
        N = len(coords_list)
    elif Z_list:
        N = len(Z_list)
    elif N_declared is not None:
        N = N_declared
    else:
        N = 0

    # If Z list missing but N known, make a default (Si=14)
    if not Z_list and N > 0:
        Z_list = [14] * N

    # If coords missing but N known, fill zeros
    if not coords_list and N > 0:
        coords_list = [[0.0, 0.0, 0.0] for _ in range(N)]

    # Defaults for lattice if missing
    if lengths is None:
        lengths = [1.0, 1.0, 1.0]
    if angles is None:
        angles = [90.0, 90.0, 90.0]

    # Tensors
    z = torch.tensor(Z_list, dtype=torch.long)
    frac_coords = torch.tensor(coords_list, dtype=torch.float32)

    if len(edges_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edges = torch.tensor(edges_list, dtype=torch.long)
        # ensure shape [2, M]
        if edges.ndim == 2 and edges.shape[1] == 2:
            edge_index = edges.t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

    # batch: per-node zeros (single structure)
    batch = torch.zeros(frac_coords.shape[0], dtype=torch.long)
    lengths_t = torch.tensor([lengths], dtype=torch.float32)
    angles_t = torch.tensor([angles], dtype=torch.float32)
    num_atoms = torch.tensor([frac_coords.shape[0]], dtype=torch.long)

    # Optional sanity checks (non-fatal)
    if N_declared is not None and N_declared != frac_coords.shape[0]:
        # You can switch to warnings.warn if you prefer
        print(f"[parse_graph] Warning: declared N={N_declared} but parsed {frac_coords.shape[0]} coords.")

    if z.numel() != frac_coords.shape[0]:
        # If mismatch, try to reconcile by trunc/pad Z
        n = frac_coords.shape[0]
        if z.numel() > n:
            z = z[:n]
        else:
            pad = torch.full((n - z.numel(),), int(z[-1].item() if z.numel() > 0 else 14), dtype=torch.long)
            z = torch.cat([z, pad], dim=0)

    return z, frac_coords, edge_index, batch, lengths_t, angles_t, num_atoms





def graph_to_text(coords: torch.Tensor, edges: torch.Tensor) -> str:
    """
    Transforms node coordinates and edge list tensors into formatted text.

    Args:
        coords (torch.Tensor): Tensor of shape (n, 3) for node coordinates.
        edges (torch.Tensor): Tensor of shape (2, m) for edges (start and end indices).

    Returns:
        str: Formatted text representation.
    """
    n = coords.size(0)
    lines = []
    lines.append(f"Node number: {n}")
    lines.append("Node coordinates:")
    for x, y, z in coords.tolist():
        lines.append(f"({x}, {y}, {z})")
    lines.append("")  # Blank line before edges
    lines.append("Edges:")
    # Assume edges is of shape (2, m)
    start_nodes, end_nodes = edges.tolist()
    for u, v in zip(start_nodes, end_nodes):
        lines.append(f"({u}, {v})")
    return "\n".join(lines)


def construct_supervisor_input(prompt, structure, lattice_lengths=None, lattice_angles=None, properties=None):
    """
    Constructs the input for the supervisor model by combining the prompt,
    structure, and optional properties.

    Args:
        prompt (str): The prompt text.
        structure (str): The structure text.
        properties (str, optional): The properties text. Defaults to None.

    Returns:
        str: The combined input string.
    """
    input_text = f"Prompt: {prompt}\n\nStructure: {structure}\n"
    if lattice_lengths is None and lattice_angles is None:
        input_text += "\nLattice lengths: [1.0, 1.0, 1.0]\nLattice angles: [90.0, 90.0, 90.0]\n"
    else:
        input_text += f"\nLattice lengths: {lattice_lengths}\nLattice angles: {lattice_angles}"
    if properties is not None:
        input_text += "\nProperties:"
        for key, value in properties.items():
            if isinstance(value, torch.Tensor):
                value = value.tolist()
            input_text += f"\n{key}: {value}"
    return input_text




def find_closest_structure(
    dataset,
    improved_properties: torch.Tensor,
    batch_size: int = 10000,   # tune for your GPU / RAM budget
    p: int = 2,                 # 2 = Euclidean, 1 = Manhattan, etc.
):
    query = torch.as_tensor(improved_properties, dtype=torch.float32).view(1, -1)

    # Make sure we run on the same device as the data (or stick to CPU)
    query = query.to(next(iter(dataset)).y.device)

    best_dist = np.inf
    best_idx  = -1

    selected_idx = []

    # Stream through the dataset in chunks
    batch = []
    start_idx = 0
    for i, data in enumerate(dataset):
        batch.append(data.y.view(-1))           # flatten to (12,)
        if len(batch) == batch_size or i == len(dataset) - 1:
            Ys = torch.stack(batch)             # (B, 12) on same device
            diff = Ys - query                   # broadcast (B, 12)
            dists = torch.linalg.vector_norm(diff, ord=p, dim=1)   # (B,)
            min_val, min_pos = torch.min(dists, dim=0)

            if min_val.item() < best_dist:
                best_dist = min_val.item()
                best_idx  = start_idx + min_pos.item()

            # reset for next chunk
            batch.clear()
            start_idx = i + 1

    return [best_idx], best_dist




