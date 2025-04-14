import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    tensor = torch.ones(1).cuda() * rank
    dist.all_reduce(tensor)
    print(f"Rank {rank} got {tensor.item()}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
