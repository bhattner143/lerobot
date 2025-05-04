import torch
import torch_scatter
import torch.nn as nn

class AttentionBlock(nn.Module):
    '''
    Basic attention block for the Updater of GAT.
    
    1. The features of each node are project to a single value using a linear layer. 
        (
            The weight is learnable, 
            the layer automatically infers the input dimension during its first use
        )
    
    2. Then the embeddings are passed through a Leaky ReLU activation function.
    
    3. Finally, the attention weights are calculated using the scatter softmax function. 
        (   
            Each calculated weight is for each incoming message (or edge) from a neighbor node.
            The idx are the indexs of receiver nodes.
        )
    
    For more details about scatter softmax, please refer to the documentation of scatter softmax:
    https://pytorch-scatter.readthedocs.io/en/1.4.0/composite/softmax.html
    '''

    def __init__(self):
        super().__init__()
        
        # Define a simple MLP to calculate the attention
        self.linear = nn.LazyLinear(1)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.mlp = nn.Sequential(
            self.linear,
            self.activation
        )

        # Define the scatter softmax function
        self.calc_weights = torch_scatter.composite.scatter_softmax

    def forward(self, input, idx):
        tmp = self.mlp(input)
        return self.calc_weights(tmp, idx, dim=1)
    

if __name__ == "__main__":
    
    # ------------------------------------------------------ #
    # Test the AttentionBlock class
    # ------------------------------------------------------ #
    # Define dummy dimensions.
    batch_size = 2
    num_neighbors = 6
    feature_dim = 10

    # Create a random input tensor with shape [batch_size, num_neighbors, feature_dim].
    input_tensor = torch.randn(batch_size, num_neighbors, feature_dim)
    
    # Define an index tensor that assigns each neighbor to a group.
    # For batch 0: first 3 neighbors belong to group 0, next 2 neighbors belong to group 1, last 1 belong to group 1.
    # For batch 1: first 2 neighbors belong to group 0, next 3 neighbors belong to group 1, last 1 belong to group 1.
    idx = torch.tensor([[0, 0, 0, 1, 1, 0],
                        [0, 0, 1, 1, 1, 0]], dtype=torch.long)
    
    print("== Dummy Input ==")
    print("Input tensor (features):")
    print(input_tensor)
    print("\nIndex tensor (group assignments):")
    print(idx)

    # Instantiate the AttentionBlock.
    attn_block = AttentionBlock()

    # Pass the dummy input data through the AttentionBlock.
    attention_weights = attn_block(input_tensor, idx)
    
    print("\n== Output ==")
    print("Attention weights (after group-wise softmax):")
    print(attention_weights)
    
    # To verify that the softmax worked per group, we print the sum of weights for each group.
    print("\nSum of attention weights per group (should be close to 1):")
    for b in range(batch_size):
        unique_groups = torch.unique(idx[b])
        for group in unique_groups:
            # Create a mask for the current group.
            group_mask = (idx[b] == group).unsqueeze(-1)  # unsqueeze to match attention_weights dims
            group_sum = attention_weights[b][group_mask].sum()
            print(f"Batch {b}, Group {group.item()} sum: {group_sum.item()}")