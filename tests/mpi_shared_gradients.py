"""Run with HMPI, 16 ranks per node, to verify both hierarchical gradient dtypes."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch import nn

from src.train.shared_parallel import (
    hierarchy_groups, share_parameters, contiguous_owners_for_model,
    HierarchicalGradients, VectorReducer,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reducer", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dist.init_process_group("mpi")
    rank, world = dist.get_rank(), dist.get_world_size()
    communication_cpu = int(os.environ["WAN21_COMMUNICATION_CPU"])
    (_, die, _, local_rank, _, local_group, owner_group,
     node_group, die_group, _) = hierarchy_groups(world, 4, 4, communication_cpu)
    args.root.mkdir(parents=True, exist_ok=True)
    reducer = VectorReducer(args.reducer)
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(45)
        model = nn.Sequential(*(nn.Linear(8, 8, bias=False) for _ in range(4))).to(dtype)
        prefix = args.root / f"gradient-gate.{str(dtype)}.die{die}"
        flat = share_parameters(model, prefix.with_suffix(prefix.suffix + ".weights"),
                                local_rank, group=local_group)
        owners = contiguous_owners_for_model(model, 4)
        gradients = HierarchicalGradients(model, prefix.with_suffix(prefix.suffix + ".gradients"),
                                         rank, world, 4, 4, communication_cpu,
                                         local_group, owner_group, node_group, die_group)
        gradients.set_owners(model, owners, 32)
        gradients.clip_norm = 1e6
        owned = [p for name, p in model.named_parameters() if owners[name] == local_rank]
        optimizer = torch.optim.AdamW(owned, lr=1e-3, fused=True)
        # Exact integers exercise accumulation, local/within-node/cross-node means,
        # split parameter buckets and the final microbatch ready hook.
        gradients.begin_step(1, reducer)
        for micro in range(2):
            gradients.publish_ready = micro == 1
            (sum(p.float().sum() for p in model.parameters()) * (rank + 1)).backward()
        norm = gradients.finish_step()
        expected = float(world + 1)
        for parameter in owned:
            torch.testing.assert_close(parameter.grad.float(),
                                       torch.full_like(parameter.grad.float(), expected), rtol=0, atol=0)
        assert abs(norm - expected * math.sqrt(256)) < 1e-4, (norm, expected)
        optimizer.step()
        dist.barrier()
        values = [None] * world
        dist.all_gather_object(values, flat.float().tolist())
        assert all(value == values[0] for value in values), "replicas disagree after owner update"
        assert all(state[key].dtype == dtype and torch.isfinite(state[key]).all()
                   for state in optimizer.state.values() for key in ("exp_avg", "exp_avg_sq"))
        if rank == 0:
            print("MPI_SHARED_GRADIENTS_OK " + json.dumps({"dtype": str(dtype), "world": world,
                                                          "grad_accum": 2, "norm": norm}), flush=True)
        dist.barrier()
        del optimizer, gradients, model, flat
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
