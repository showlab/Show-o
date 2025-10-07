import torch


def count_by_gate(gate, num_expert, world_size, require_pos=True):
    """
    Count tokens assigned to each expert.
    
    Args:
        gate: 1-D tensor of expert indices for each token
        num_expert: number of experts
        world_size: world size (for distributed, currently only supports 1)
        require_pos: whether to compute positions
    
    Returns:
        pos: positions for each token (if require_pos=True, else None)
        local_expert_count: count of tokens for each expert locally
        global_expert_count: count of tokens for each expert globally
    """
    with torch.no_grad():
        # Count how many tokens go to each expert
        local_expert_count = torch.zeros(
            num_expert * world_size, device=gate.device, dtype=torch.long
        )
        
        # Count tokens for each expert
        valid_gate = gate[gate >= 0]  # Filter out -1 (dropped tokens)
        if valid_gate.numel() > 0:
            local_expert_count.scatter_add_(
                0, valid_gate, torch.ones_like(valid_gate, dtype=torch.long)
            )
        
        if world_size > 1:
            # For distributed training, would need all_to_all communication
            # For now, just support single GPU
            raise NotImplementedError("Distributed training (world_size > 1) not supported")
        else:
            global_expert_count = local_expert_count
        
        if not require_pos:
            pos = None
        else:
            # Compute cumulative sum to assign positions
            lec_cum = torch.cumsum(local_expert_count, dim=0)
            pos_size = lec_cum[-1].item()
            pos = torch.empty((pos_size,), device=gate.device, dtype=torch.long)
            
            # Assign position to each token based on its expert
            # This is a simplified version
            for i in range(len(gate)):
                if gate[i] >= 0:
                    expert_id = gate[i].item()
                    if expert_id > 0:
                        offset = lec_cum[expert_id - 1].item()
                    else:
                        offset = 0
                    # Find position within this expert's allocation
                    count_before = (gate[:i] == gate[i]).sum().item()
                    pos[offset + count_before] = i
                    
    return pos, local_expert_count.int(), global_expert_count.int()


def limit_by_capacity(topk_idx, num_expert, world_size, capacity):
    """
    Limit the number of tokens that can be routed to each expert.
    
    Args:
        topk_idx: tensor of expert indices [batch_size, top_k]
        num_expert: number of experts
        world_size: world size for distributed
        capacity: maximum capacity per expert
    
    Returns:
        new_lec: new local expert count
        new_gec: new global expert count  
        topk_idx: pruned topk_idx with -1 for dropped tokens
    """
    with torch.no_grad():
        # Flatten topk_idx to 1D
        topk_idx_flat = topk_idx.contiguous().view(-1)
        
        # Count tokens per expert
        _, lec, gec = count_by_gate(topk_idx_flat, num_expert, world_size, require_pos=False)
        
        # Limit by capacity
        capacity_tensor = torch.ones(num_expert * world_size, dtype=torch.int32,
                device=topk_idx.device) * capacity
        
        new_gec = torch.minimum(gec.to(torch.int32), capacity_tensor)
        
        if world_size > 1:
            raise NotImplementedError("Distributed training (world_size > 1) not supported")
        else:
            new_lec = new_gec
        
        # Prune tokens that exceed capacity
        topk_idx_pruned = prune_gate_by_capacity(
            topk_idx_flat, new_lec, num_expert, world_size
        )
        
        topk_idx_pruned = topk_idx_pruned.view_as(topk_idx)
        
    return new_lec, new_gec, topk_idx_pruned


def prune_gate_by_capacity(gate, capacity, num_expert, world_size):
    """
    Prune gate assignments that exceed expert capacity.
    Tokens that exceed capacity are marked with -1.
    
    Args:
        gate: tensor of expert indices (can be 1-D or 2-D)
        capacity: capacity per expert
        num_expert: number of experts
        world_size: world size
        
    Returns:
        pruned_gate: gate with excess tokens set to -1
    """
    with torch.no_grad():
        original_shape = gate.shape
        gate_flat = gate.contiguous().view(-1)
        pruned_gate = gate_flat.clone()
        
        # Count how many tokens we've assigned to each expert so far
        expert_count = torch.zeros(num_expert * world_size, 
                                   device=gate.device, dtype=torch.int32)
        
        # Go through each token and check if expert has capacity
        for i in range(len(gate_flat)):
            expert_id = gate_flat[i].item()
            if expert_id >= 0:  # Valid expert ID
                if expert_count[expert_id] < capacity[expert_id]:
                    expert_count[expert_id] += 1
                else:
                    # Exceeded capacity, drop this token
                    pruned_gate[i] = -1
        
        # Restore original shape
        pruned_gate = pruned_gate.view(original_shape)
                    
    return pruned_gate
