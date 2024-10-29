import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def sample_from_logits(logits, temperature=1.0, top_k=0, top_p=0.5):
    """
    Adjusted from:
        - https://github.com/google-research/google-research/tree/master/multi_game_dt
        - https://github.com/etaoxing/multigame-dt
    """
    logits = logits.double()
    if top_p > 0.0:
        # percentile: 0 to 100, quantile: 0 to 1
        # torch.quantile cannot handle float16
        percentile = torch.quantile(logits, top_p, dim=-1)
        if percentile != logits.max():
            # otherwise all logits would become -inf
            logits = torch.where(logits > percentile.unsqueeze(-1), logits, -float("inf"))
    if top_k > 0:
        logits, top_indices = torch.topk(logits, top_k)
    sample = torch.distributions.Categorical(logits=temperature * logits).sample()
    if top_k > 0:
        sample_shape = sample.shape
        # Flatten top-k indices and samples for easy indexing.
        top_indices = torch.reshape(top_indices, [-1, top_k])
        sample = sample.flatten()
        sample = top_indices[torch.arange(len(sample)), sample]
        # Reshape samples back to original dimensions.
        sample = torch.reshape(sample, sample_shape)
    return sample


def position_encoding_init(n_position, d_pos_vec):
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
            for pos in range(n_position)])

    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def make_sinusoidal_embd(n_positions, embed_dim): 
    position_enc = torch.nn.Embedding(n_positions, embed_dim)
    position_enc.weight.data = position_encoding_init(n_positions, embed_dim)
    return position_enc


def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


class SwiGLU(nn.Module):
    # SwiGLU https://arxiv.org/abs/2002.05202
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class GEGLU(nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    """

    def geglu(self, x):
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

    def forward(self, x):
        return self.geglu(x)
