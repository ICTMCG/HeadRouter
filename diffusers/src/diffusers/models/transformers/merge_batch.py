import torch
from typing import Tuple, Callable


def do_nothing(x: torch.Tensor, mode:str=None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def bipartite_soft_matching_batch_split(metric: torch.Tensor,
                                        w: int, h: int, sx: int, sy: int, r: int,
                                        no_rand: bool = False,
                                        generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst based on batch dimension: 
    first batch is src, second batch is dst.
    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - generator: random generator
    """
    B, N, _ = metric.shape

    if B != 2:
        raise ValueError("The batch size must be 2 to split into src and dst.")

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # We don't need random index generation anymore since we split by batch
        # Directly assign first batch as src and second as dst
        a_idx = torch.arange(N, device=metric.device).expand(1, N, 1)  # src is batch 0
        b_idx = torch.arange(N, device=metric.device).expand(1, N, 1)  # dst is batch 1

        def split(x):
            C = x.shape[-1]
            src = x[0]  # src from batch 0
            dst = x[1]  # dst from batch 1
            return src, dst

        # Normalize the metric to compute cosine similarity
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[0], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        t1, c = src.shape
        
        unm = gather(src, dim=-2, index=unm_idx.expand(t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=0).unsqueeze(0)  # Return with batch dimension

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(r, c))

        # Combine back to the original shape
        out = torch.zeros(2, N, c, device=x.device, dtype=x.dtype)
        out[0] = torch.cat([unm, src], dim=0)  # src tokens
        out[1] = dst  # dst tokens

        return out

    return merge, unmerge
