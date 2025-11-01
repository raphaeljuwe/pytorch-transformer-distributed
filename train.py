import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from model import TransformerModel, DummyDataset

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] Process group initialized with NCCL backend")

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    print(f"Starting training on rank {rank}/{world_size}")
    setup(rank, world_size)

    # Model, dataset, and dataloader setup
    model = TransformerModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    dataset = DummyDataset(size=1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(5):
        sampler.set_epoch(epoch)
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(rank), y.to(rank)
            optimizer.zero_grad()
            output = ddp_model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices available. Please attach a GPU to your Studio.")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
