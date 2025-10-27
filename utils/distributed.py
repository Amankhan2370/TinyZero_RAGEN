import os
import torch
import torch.distributed as dist
from typing import Dict, Any, List
import numpy as np

def setup_distributed() -> int:
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        
        print(f"Initialized distributed training: "
              f"rank {rank}, world_size {world_size}, local_rank {local_rank}")
        return local_rank
    else:
        # Single GPU training
        return 0

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_world_size() -> int:
    """Get the number of processes in the distributed group."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

def get_rank() -> int:
    """Get the rank of the current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0

def all_reduce_dict(stats_dict: Dict[str, Any]) -> Dict[str, Any]:
    """All-reduce a dictionary of statistics across processes."""
    if not dist.is_initialized():
        return stats_dict
    
    # Convert values to tensors
    tensor_dict = {}
    for key, value in stats_dict.items():
        if isinstance(value, (int, float)):
            tensor_dict[key] = torch.tensor(float(value), device='cuda')
        elif isinstance(value, torch.Tensor):
            tensor_dict[key] = value.cuda()
        else:
            # Skip non-numeric values
            continue
    
    # All-reduce each tensor
    for key in tensor_dict:
        dist.all_reduce(tensor_dict[key])
        tensor_dict[key] = tensor_dict[key] / get_world_size()
    
    # Convert back to original types
    result_dict = stats_dict.copy()
    for key, tensor in tensor_dict.items():
        original_value = stats_dict[key]
        if isinstance(original_value, int):
            result_dict[key] = int(tensor.item())
        else:
            result_dict[key] = tensor.item()
    
    return result_dict

def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from source to all processes."""
    if dist.is_initialized():
        dist.broadcast(tensor, src)
    return tensor

def gather_tensors(tensor: torch.Tensor) -> List[torch.Tensor]:
    """Gather tensors from all processes."""
    if not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return tensor_list

def sync_barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()

def setup_seed(seed: int):
    """Set random seed for reproducibility across processes."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False