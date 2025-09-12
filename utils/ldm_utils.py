import torch
import torch.nn.functional as F


# -------------- Utils for LDM --------------
def convert_edge_list_to_edge_index(edge_list, num_nodes):
    """
    Convert an edge list to edge index format.
    :param edge_list: List of batch edges, where each edge is a tuple (source, target).
    :param num_nodes: Total number of nodes in the graph.
    :return: Edge index tensor of shape [2, num_edges].
    """
    offset = 0
    new_edge_list = []
    for i, num_node in enumerate(num_nodes):
        edge_tensor_i = edge_list[i]+offset
        new_edge_list.append(edge_tensor_i)
        offset += num_node
    edge_index = torch.cat(new_edge_list, dim=-1)  # shape=[2, num_edges]
    return edge_index



def get_node_mask(node_num_pred, max_batch_num_node):
    b = node_num_pred.shape[0]
    index_range = torch.arange(max_batch_num_node, device=node_num_pred.device)
    index_batched = index_range.unsqueeze(0).expand(b, -1)
    mask_2d = index_batched < node_num_pred.unsqueeze(1)
    node_mask = mask_2d.unsqueeze(-1)
    node_mask = node_mask.to(dtype=torch.bool)
    return node_mask

def compute_edge_bce_loss_with_pos_weight(
        logits,
        node_mask,
        edge_index,
        num_edges,
        is_weight=True
):
    device = logits.device
    B, N, _ = logits.shape
    node_mask = node_mask.squeeze(-1)

    target_adj = torch.zeros((B, N, N), dtype=torch.float, device=device)

    edge_offset = 0
    node_offset = 0
    for b in range(B):
        ecount = num_edges[b]
        ncount = node_mask[b].sum()
        if ecount == 0:
            # 没有边, 跳过
            continue

        # 全局范围 [edge_offset, edge_offset + ecount)
        this_edge_range = slice(edge_offset, edge_offset + ecount)
        this_edge_idx = edge_index[:, this_edge_range]  # shape=[2, ecount]

        global_row = this_edge_idx[0]  # shape=[ecount]
        global_col = this_edge_idx[1]  # shape=[ecount]

        local_row = global_row - node_offset  # => [0..num_atoms_b-1]
        local_col = global_col - node_offset


        target_adj[b, local_row, local_col] = 1.0

        edge_offset += ecount
        node_offset += ncount

    row_idx, col_idx = torch.triu_indices(N, N, offset=1, device=device)  # 仅上三角

    pair_mask = torch.zeros((B, N, N), dtype=torch.bool, device=device)

    for b in range(B):
        # node_mask[b]: [N], True/False
        valid_j = node_mask[b, row_idx]  # shape=[M]
        valid_i = node_mask[b, col_idx]  # shape=[M]
        valid_pairs = (valid_j & valid_i).to(device)  # 同时为 True => 这一对节点有效
        pair_mask[b, row_idx[valid_pairs], col_idx[valid_pairs]] = True

    valid_logits = logits[pair_mask]  # [num_valid_pairs]
    valid_target = target_adj[pair_mask]  # [num_valid_pairs]

    if valid_logits.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    if is_weight:
        num_pos = valid_target.sum()
        num_neg = valid_target.numel() - num_pos
        if num_pos == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        pos_weight_val = num_neg / num_pos
        pos_weight = pos_weight_val.clone().detach()

        loss = F.binary_cross_entropy_with_logits(
            valid_logits, valid_target, pos_weight=pos_weight, reduction='mean'
        )
    else:
        loss = F.binary_cross_entropy_with_logits(
            valid_logits, valid_target
        )
    return loss



def padding_graph2max_num_node(x, batch, max_num_node):
    """

    :param x: node feature [all node num, feat_dim]
    :param batch: graph indices wrt node
    :param max_num_node:
    :return:
        padded_x=[batch_size, max_num_node, feat_dim]
        mask=[batch_size, max_num_node, 1]
    """
    device = x.device
    num_nodes, feat_dim = x.shape

    batch_size = batch[-1].item() + 1

    # 1.
    graph_indices = torch.arange(batch_size, device=device)
    left_boundary = torch.searchsorted(batch, graph_indices, right=False)
    # eg. batch=[0,0,0,1,1,2] => left_boundary=[0,3,5]

    # 2.
    i = torch.arange(num_nodes, device=device)
    offset_in_graph = i - left_boundary[batch]

    # 3.
    index = batch * max_num_node + offset_in_graph

    # 4
    padded = x.new_zeros(batch_size * max_num_node, feat_dim)
    padded[index] = x

    # 5
    padded_x = padded.view(batch_size, max_num_node, feat_dim)

    mask_1d = torch.zeros(batch_size * max_num_node, dtype=torch.bool, device=device)
    mask_1d[index] = True
    mask = mask_1d.view(batch_size, max_num_node,1)
    return padded_x, mask


def unpadding_max_num_node2graph(padded_x, mask):
    device = padded_x.device
    batch_size, max_num_node, feat_dim = padded_x.shape

    padded_x_flat = padded_x.view(-1, feat_dim)  # shape = [B*N, feat_dim]

    mask_flat = mask.view(-1)  # shape = [B*N]

    valid_idx = torch.nonzero(mask_flat, as_tuple=True)[0]  # 1D 下标

    x = padded_x_flat[valid_idx]  # shape = [num_real_nodes, feat_dim]

    graph_idx_2d = torch.arange(batch_size, device=device).unsqueeze(1)
    graph_idx_2d = graph_idx_2d.expand(batch_size, max_num_node)  # shape=[B,N]

    batch_flat = graph_idx_2d.flatten()  # shape=[B*N]
    batch = batch_flat[valid_idx]  # shape=[num_real_nodes]

    return x, batch



def farthest_point_sampling(x, num_samples):
    """
    x: [N, D] node embeddings
    return: indices of sampled nodes [num_samples]
    """
    N, D = x.shape
    selected = [torch.randint(0, N, (1,)).item()]
    distances = torch.full((N,), float('inf'), device=x.device)

    for _ in range(1, num_samples):
        last = x[selected[-1]].unsqueeze(0)             # [1,D]
        dist = ((x - last)**2).sum(dim=1)                # [N]
        distances = torch.min(distances, dist)
        selected.append(distances.argmax().item())

    return torch.tensor(selected, device=x.device)
