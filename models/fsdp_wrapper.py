import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from typing import Dict, Any

class FSDPConfig:
    """Configuration for Fully Sharded Data Parallel training."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.sharding_strategy = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }[config_dict.get("sharding_strategy", "FULL_SHARD")]
        
        self.mixed_precision = config_dict.get("mixed_precision", True)
        self.gradient_checkpointing = config_dict.get("gradient_checkpointing", True)
        self.cpu_offload = config_dict.get("cpu_offload", False)
        
    def get_mixed_precision_policy(self) -> MixedPrecision:
        """Get mixed precision policy for FSDP."""
        if self.mixed_precision:
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        else:
            return MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )

def get_fsdp_policy(transformer_block_class=None):
    """Get FSDP wrapping policy for model layers."""
    if transformer_block_class:
        return transformer_auto_wrap_policy(
            transformer_layer_cls={transformer_block_class}
        )
    else:
        return size_based_auto_wrap_policy(min_num_params=1000000)

def setup_fsdp_model(model: nn.Module, fsdp_config: FSDPConfig) -> FSDP:
    """Wrap model with FSDP for distributed training."""
    
    # Enable gradient checkpointing if specified
    if fsdp_config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Configure CPU offload
    cpu_offload = CPUOffload(offload_params=fsdp_config.cpu_offload)
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=get_fsdp_policy(TransformerBlock),
        mixed_precision=fsdp_config.get_mixed_precision_policy(),
        sharding_strategy=fsdp_config.sharding_strategy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=cpu_offload,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
    )
    
    return model

def save_fsdp_model(model: FSDP, path: str):
    """Save FSDP model checkpoint."""
    # Gather model state dict from all ranks
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()
        
    if torch.distributed.get_rank() == 0:
        torch.save(state_dict, path)

def load_fsdp_model(model: FSDP, path: str):
    """Load FSDP model checkpoint."""
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    
    # Load state dict on rank 0 and broadcast
    if torch.distributed.get_rank() == 0:
        state_dict = torch.load(path, map_location="cpu")
    else:
        state_dict = None
        
    state_dict = torch.distributed.broadcast(state_dict, src=0)
    
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
        model.load_state_dict(state_dict)