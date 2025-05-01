import torch

# Check if torch_scatter is available
TORCH_SCATTER_AVAILABLE = False
try:
    import torch_scatter
    TORCH_SCATTER_AVAILABLE = True
except ImportError as e:
    print(f"torch_scatter is not installed: {e}. Falling back to custom_scatter_softmax.")

import torch.nn as nn

def custom_scatter_softmax(src, index, dim=-1):
    """
    Custom implementation of scatter_softmax.
    
    Args:
        src (Tensor): The input tensor containing the values to apply softmax to.
        index (Tensor): The index tensor defining the groups.
        dim (int): The dimension along which to apply the softmax.

    Returns:
        Tensor: The tensor with softmax applied within each group.
    """
    if dim < 0:
        dim += src.dim()
    expanded_index = index.unsqueeze(-1).expand_as(src)

    max_per_group = torch.zeros_like(src).scatter_reduce_(
        dim, expanded_index, src, reduce="amax", include_self=False
    )

    src_exp = src - max_per_group.gather(dim, expanded_index)
    exp_per_group = src_exp.exp()

    sum_exp_per_group = torch.zeros_like(src).scatter_reduce_(
        dim, expanded_index, src, reduce="sum", include_self=False
    )

    softmax_per_group = exp_per_group / sum_exp_per_group.gather(dim, expanded_index)
    return softmax_per_group

class AttentionBlock(nn.Module):
    '''
    Basic attention block for the Updater of GAT.
    '''

    def __init__(self):
        super().__init__()
        
        self.linear = nn.LazyLinear(1)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.mlp = nn.Sequential(
            self.linear,
            self.activation
        )

        # Use torch_scatter if available, otherwise fallback to custom implementation
        if TORCH_SCATTER_AVAILABLE:
            self.calc_weights = torch_scatter.composite.scatter_softmax
        else:
            self.calc_weights = custom_scatter_softmax

    def forward(self, input, idx):
        tmp = self.mlp(input)
        return self.calc_weights(tmp, idx, dim=1)
    

if __name__ == "__main__":
    batch_size = 2
    num_neighbors = 6
    feature_dim = 10

    input_tensor = torch.randn(batch_size, num_neighbors, feature_dim)
    idx = torch.tensor([[0, 0, 0, 1, 1, 0],
                        [0, 0, 1, 1, 1, 0]], dtype=torch.long)
    
    print("== Dummy Input ==")
    print("Input tensor (features):")
    print(input_tensor)
    print("\nIndex tensor (group assignments):")
    print(idx)

    attn_block = AttentionBlock()
    attention_weights = attn_block(input_tensor, idx)
    
    print("\n== Output ==")
    print("Attention weights (after group-wise softmax):")
    print(attention_weights)
    
    print("\nSum of attention weights per group (should be close to 1):")
    for b in range(batch_size):
        unique_groups = torch.unique(idx[b])
        for group in unique_groups:
            group_mask = (idx[b] == group).unsqueeze(-1)
            group_sum = attention_weights[b][group_mask].sum()
            print(f"Batch {b}, Group {group.item()} sum: {group_sum.item()}")
