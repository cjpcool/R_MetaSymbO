import torch
import torch.nn.functional as F

# ---------- 1.1  pairwise cost (L2) ----------
def pairwise_l2(a, b):
    """
    a : [N_s, D]   node embeddings  (mu  or raw features)
    b : [N_t, D]
    return  cost : [N_s, N_t]
    """
    a2 = (a**2).sum(-1, keepdim=True)         # [N_s,1]
    b2 = (b**2).sum(-1, keepdim=True).T       # [1,N_t]
    return a2 + b2 - 2 * a @ b.T              # broadcasting

# ---------- 1.2  Sinkhorn ----------
def sinkhorn(cost, epsilon=0.1, max_iter=50, tol=1e-6):
    """
    cost : [N_s, N_t] non‑negative matrix
    returns P (transport plan)  [N_s, N_t]
    """
    N_s, N_t = cost.shape
    u = torch.ones(N_s, device=cost.device) / N_s
    v = torch.ones(N_t, device=cost.device) / N_t
    K = torch.exp(-cost / epsilon)            # Gibbs kernel

    for _ in range(max_iter):
        u_prev = u.clone()
        u = 1. / (K @ v)
        v = 1. / (K.T @ u)
        if torch.max(torch.abs(u - u_prev)) < tol:
            break

    P = torch.diag(u) @ K @ torch.diag(v)     # [N_s,N_t]
    return P                  


def sinkhorn_log(cost, epsilon=0.1, max_iter=50, tol=1e-6, verbose=False):
    """
    cost : [N_s, N_t]
    returns P: [N_s, N_t]
    """
    N_s, N_t = cost.shape
    log_K = -cost / epsilon                 # [N_s, N_t]
    
    f = torch.zeros(cost.shape[0], device=cost.device)
    g = torch.zeros(cost.shape[1], device=cost.device)

    for i in range(max_iter):
        f_prev = f.clone()

        # numerically stabilized row logsumexp
        log_K1 = log_K + g[None, :]                     # [N_s, N_t]
        log_K1 = log_K1 - log_K1.max(dim=1, keepdim=True)[0]
        f = -torch.logsumexp(log_K1, dim=1)

        # stabilized col logsumexp
        log_K2 = log_K + f[:, None]
        log_K2 = log_K2 - log_K2.max(dim=0, keepdim=True)[0]
        g = -torch.logsumexp(log_K2, dim=0)

        delta = (f - f_prev).abs().max().item()
        if verbose:
            print(f"iter {i}: max Δf = {delta:.5f}")
        if delta > 1e4:  # NaN-safe guard
            print("Sinkhorn diverged at iter", i)
            break
        if delta < tol:
            break

    # transport plan
    log_P = log_K + f[:, None] + g[None, :]
    P = torch.exp(log_P)
    return P



 # ---------- utils ----------
def to_sigma2(logvar, eps=1e-8):
    return (logvar).exp().clamp_min(eps)       # σ²  >= eps

def kl_gaussian_diag(mu, logvar, mu_t, logvar_t, eps=1e-8):
    var    = to_sigma2(logvar,    eps)
    var_t  = to_sigma2(logvar_t,  eps)
    kl = 0.5 * (logvar_t - logvar +
                (var + (mu - mu_t).pow(2)) / var_t - 1)
    return kl.sum(-1)                          # no mean ‑ easier to re‑weight







def gaussian_negation(mu_M, logvar_M, mu_Mp, logvar_Mp, alpha=1.0, beta=0.5, eps=1e-6):
    # Convert logvar to precision (inverse of diagonal covariance)
    precision_M = torch.exp(-logvar_M)
    precision_Mp = torch.exp(-logvar_Mp)

    # Compute negated precision
    precision_neg = alpha * precision_M - beta * precision_Mp

    # Ensure positivity (for numerical stability)
    precision_neg = torch.clamp(precision_neg, min=eps)
    var_neg = 1.0 / precision_neg
    logvar_neg = torch.log(var_neg)

    # Compute negated mean
    mu_neg = var_neg * (alpha * precision_M * mu_M - beta * precision_Mp * mu_Mp)

    return mu_neg, logvar_neg






# Product-of-Experts (PoE) for Gaussian distributions, Intersection of two Gaussians
def poe(mu_a, logvar_a, mu_b, logvar_b, detach_a=True, detach_b=True, eps=1e-8):
    if detach_a:  #  True
        mu_a, logvar_a = mu_a.detach(), logvar_a.detach()
    if detach_b:  #  True
        mu_b, logvar_b = mu_b.detach(), logvar_b.detach()
    var_a, var_b = to_sigma2(logvar_a, eps), to_sigma2(logvar_b, eps)
    tau = 1./var_a + 1./var_b
    mu = (mu_a/var_a + mu_b/var_b) / tau
    logvar = (-torch.log(tau)).clamp(-30., 30.)
    return mu, logvar
# Quatient-of-Experts (QoE) for Gaussian distributions, difference of two Gaussians
def qoe(mu_a, logvar_a, mu_b, logvar_b, detach_a=True, detach_b=True, eps=1e-8):
    if detach_a:  #  True
        mu_a, logvar_a = mu_a.detach(), logvar_a.detach()
    if detach_b:  #  True
        mu_b, logvar_b = mu_b.detach(), logvar_b.detach()
    var_a, var_b = to_sigma2(logvar_a, eps), to_sigma2(logvar_b, eps)
    tau = (1.0 / var_a) - (1.0 / var_b)                # τ = τ_a - τ_b
    # 安全：若出现负precision，回退到 tiny‑precision
    tau = torch.clamp(tau, min=eps)
    mu  = ( (mu_a/var_a) - (mu_b/var_b) ) / tau
    logvar = (-torch.log(tau)).clamp(min=-30., max=30.)
    return mu, logvar


# Moment-Matching (MM) for Gaussian distributions, mixture of two Gaussians
def mixture_mm(mu_a, logvar_a, mu_b, logvar_b, detach_a=False, detach_b=True, lam=0.5, eps=1e-8):
    if detach_a:  #  False
        mu_a, logvar_a = mu_a.detach(), logvar_a.detach()
    if detach_b:  #  True
        mu_b, logvar_b = mu_b.detach(), logvar_b.detach()
    var_a, var_b = to_sigma2(logvar_a, eps), to_sigma2(logvar_b, eps)
    mu = (1-lam)*mu_a + lam*mu_b
    sigma = (1-lam)*var_a.sqrt() + lam*var_b.sqrt()
    logvar = (sigma**2 + eps).log()
    return mu, logvar


def node_logic_loss(P,
                    mu_a, logvar_a,
                    mu_b, logvar_b,
                    logic_mode="mix",
                    lam=0.5,
                    detach_b=True,
                    thr=0.1,
                    eps=1e-8):
    """
    · mu_a/logvar_a : [N_s, D]
    · mu_b/logvar_b : [N_t, D]
    · P             : [N_s, N_t]  soft matching
    """
    N_s, N_t, D = mu_a.shape[0], mu_b.shape[0], mu_a.shape[1]

    # ---------- broadcast ----------
    mu_a_ = mu_a[:, None, :].expand(N_s, N_t, D)
    lv_a_ = logvar_a[:, None, :].expand_as(mu_a_)
    mu_b_ = mu_b[None, :, :].expand(N_s, N_t, D)
    lv_b_ = logvar_b[None, :, :].expand_as(mu_b_)

    # ---------- symbolic ----------
    if logic_mode == "int":
        mu_t, lv_t = poe(mu_a_, lv_a_, mu_b_, lv_b_, eps=eps)
    elif logic_mode == "neg":
        # mu_t, lv_t = qoe(mu_a_, lv_a_, mu_b_, lv_b_, eps=eps)
        mu_t, lv_t = gaussian_negation(mu_a_, lv_a_, mu_b_, lv_b_,
                                       alpha=1.0, beta=0.01, eps=eps)
    elif logic_mode == "mix":
        mu_t, lv_t = mixture_mm(mu_a_, lv_a_, mu_b_, lv_b_,
                                lam=lam, eps=eps)
    else:
        raise ValueError
    
    row_mass = P.sum(1)                     # [N_s]
    col_mass = P.sum(0)                     # [N_t]
    
    # only compute overlapping nodes
    matched_a = (row_mass > thr)            # [N_s] bool
    matched_b = (col_mass > thr)            # [N_t] bool
    # P = P[matched_a]
    # mu_a_ = mu_a_[matched_a]                # [N_s, N_t, D]
    # lv_a_ = lv_a_[matched_a]                # [N_s, N_t, D]
    # mu_t  = mu_t[matched_a]                 # [N_s, D]
    # lv_t  = lv_t[matched_a]                 # [N_s, D]
    selected_idx = P.argmax(dim=1)   
    r = torch.arange(N_s, device=mu_a.device)  # [N_s] index
    
    kl_pair = kl_gaussian_diag(mu_a_, lv_a_, mu_t, lv_t, eps=eps)  # [N_s,N_t]

    L_match = (P[r, selected_idx] * kl_pair[r, selected_idx]).sum()



    return L_match

def node_logic_keep_original(P, mu_a_new, logvar_a_new, mu_a_old, logvar_a_old, 
                             thr=0.1, eps=1e-8):
    row_mass = P.sum(1)                       # [N_s]
    # col_mass = P.sum(0)                       # [N_t]
        
    keep_a = (row_mass <= thr)
    # keep_b = (col_mass <= thr)
    if keep_a.any():
        kl_a = kl_gaussian_diag(mu_a_new[keep_a], logvar_a_new[keep_a], mu_a_old[keep_a].detach(), logvar_a_old[keep_a].detach(), eps=eps).sum()
    else:
        kl_a = torch.tensor(0.0, device=mu_a_new.device)
    # if keep_b.any():
    #     kl_b = kl_gaussian_diag(mu_b_new[keep_b], logvar_b_new[keep_b], mu_b_old[keep_b].detach(), logvar_b_old[keep_b].detach(), eps=eps).sum()
    return kl_a

def nodes_to_keep(P,
                  logic_mode="int",
                  thresh=0.05):
    """
    P          : [N_s, N_t]，Sinkhorn soft‑matching
    logic_mode : "poe" | "qoe" | "mix"
    thresh     : 
    ------------------------------------------
    return
      keep_src : [N_s] bool  ——
      keep_tgt : [N_t] bool  ——
    """
    row_mass = P.sum(1)                       # [N_s]
    col_mass = P.sum(0)                       # [N_t]

    if logic_mode == "int":                   # Intersection keep source nodes
        keep_src = row_mass > thresh
        keep_tgt = torch.zeros_like(col_mass, dtype=torch.bool)

    elif logic_mode == "neg":                 # difference keep source - target nodes
        keep_src = row_mass <= thresh         #
        keep_tgt = torch.zeros_like(col_mass, dtype=torch.bool)

    elif logic_mode == "mix" or logic_mode == 'union':                 # union: keep source + alone target nodes
        keep_src = torch.ones_like(row_mass, dtype=torch.bool)
        keep_tgt = col_mass <= thresh         # include alone target nodes

    else:
        raise ValueError("logic_mode must be int/neg/mix")

    return keep_src, keep_tgt




def gather_nodes_and_edges(
        coords_src, edge_index_src,
        coords_tgt, edge_index_tgt,
        P,                       # [N_s, N_t] Sinkhorn plan
        logic_mode="int",
        thresh=0.05):
    """
    Merge two graphs (source / target) according to symbolic‑logic rule.

    Parameters
    ----------
    coords_src : Tensor [N_s, 3]
    edge_index_src : LongTensor [2, E_s]
    coords_tgt : Tensor [N_t, 3]
    edge_index_tgt : LongTensor [2, E_t]
    P : Tensor [N_s, N_t]  — Sinkhorn transport
    thresh : float — matching prob threshold for overlapping nodes

    Returns
    -------
    coords_all : Tensor [N', 3]
    edge_index_all : LongTensor [2, E']
    """

    keep_src, keep_tgt = nodes_to_keep(P, logic_mode, thresh)   # bool masks
    N_s, N_t = keep_src.size(0), keep_tgt.size(0)
    offset   = N_s                                             # tgt 节点全体 +offset

    coords_all = coords_src[keep_src]
    if logic_mode in ('mix',"union") and keep_tgt.any():
        coords_all = torch.cat([coords_all, coords_tgt[keep_tgt]], dim=0)

    old2new = torch.full((N_s + N_t,), -1, dtype=torch.long,
                         device=coords_src.device)

    new_id = 0
    for i in range(N_s):
        if keep_src[i]:
            old2new[i] = new_id
            new_id += 1

    best_src_for_tgt = P.argmax(dim=0)            # [N_t]
    for j in range(N_t):
        tgt_old = offset + j                      # 原全局 id
        if keep_tgt[j]:                           # 情况 (a)
            old2new[tgt_old] = new_id
            new_id += 1
        else:                                     # 情况 (b)  — 重合节点
            src_match = best_src_for_tgt[j].item()
            if keep_src[src_match]:
                old2new[tgt_old] = old2new[src_match]  # 指向已保留的源节点

    edge_components = [edge_index_src.clone()]
    if logic_mode in ( 'mix',"union"):
        edge_components.append(edge_index_tgt.clone() + offset)
    edge_tot = torch.cat(edge_components, dim=1)          # [2, E_tot]

    idx0 = old2new[edge_tot[0]]
    idx1 = old2new[edge_tot[1]]

    valid = (idx0 >= 0) & (idx1 >= 0) & (idx0 != idx1)
    edge_index_all = torch.stack([idx0[valid], idx1[valid]], dim=0)

    return coords_all, edge_index_all

def check_all_nodes_connected(coords_all, edge_index_all):
    # check if each nodes have at least one edge
    for i in range(len(coords_all)):
        if (edge_index_all[0] == i).sum() == 0 and (edge_index_all[1] == i).sum() == 0:
            print(f"Node {i} has no edges.")
            coords_all = torch.cat([coords_all[:i], coords_all[i+1:]])
            mask_i = ((edge_index_all[0] == i) | (edge_index_all[1] == i))
            edge_index_all = edge_index_all[:, ~mask_i]
            edge_index_all[edge_index_all > i] = edge_index_all[edge_index_all > i] - 1

    return coords_all, edge_index_all

def remove_close_nodes(coords_all, edge_index_all, min_dist=0.1):
    """
    Remove nodes that are too close to each other (within min_dist).
    Keeps the first occurrence and removes duplicates within the distance threshold.
    
    Args:
        coords_all (torch.Tensor): Tensor of shape [N, 3] for node coordinates.
        edge_index_all (torch.Tensor): Tensor of shape [2, E] for edges.
        min_dist (float): Minimum distance to keep nodes separate.

    Returns:
        coords_all (torch.Tensor): Filtered coordinates.
        edge_index_all (torch.Tensor): Updated edge index.
    """
    N = coords_all.size(0)
    keep_mask = torch.ones(N, dtype=torch.bool).to(coords_all.device)
    
    for i in range(N):
        if not keep_mask[i]:
            continue
        dists = torch.norm(coords_all[i] - coords_all, dim=1)
        too_close = (dists < min_dist) & (torch.arange(N, device=coords_all.device) > i)
        keep_mask[too_close] = False

    # Mapping from old indices to new indices
    new_idx = torch.cumsum(keep_mask, dim=0) - 1
    coords_all = coords_all[keep_mask]

    # Filter and remap edge indices
    mask_edge = keep_mask[edge_index_all[0]] & keep_mask[edge_index_all[1]]
    edge_index_all = edge_index_all[:, mask_edge]
    edge_index_all = new_idx[edge_index_all]

    return coords_all, edge_index_all