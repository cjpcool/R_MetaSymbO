import torch



def pooling_gaussian(mu, logvar):
    pool_weight = torch.ones_like(mu[...,0]) 
    mu_pool  = (mu * pool_weight.unsqueeze(-1)).sum(0) / pool_weight.sum(0)
    second_moment = ((mu**2 + logvar.exp()) * pool_weight.unsqueeze(-1)).sum(0) / pool_weight.sum(0)
    var_pool = second_moment - mu_pool**2
    logvar_pool = var_pool.clamp_min(1e-8).log()
    return mu_pool, logvar_pool

