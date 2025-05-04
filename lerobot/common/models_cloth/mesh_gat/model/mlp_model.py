import torch
import collections
import torch.nn as nn

class LazyMLPBlock(nn.Module):
    """
        Basic MLP structure.
        We use LazyLinear to define the linear layer, and ReLU as the activation function.
        
        output_sizes: list of integers, the output size of each layer.
            for example, [100, 200, 300] means three layers have 100/200/300 dimension output, respectively.
    """
    
    def __init__(self, output_sizes):
        super(LazyMLPBlock, self).__init__()

        # Get number of layers based on provided output sizes.
        num_layers = len(output_sizes)

        # Construct linear-ReLU MLP layers using an OrderedDict.
        self._layers_ordered_dict = collections.OrderedDict()
        for index, output_size in enumerate(output_sizes):
            self._layers_ordered_dict["linear_" + str(index)] = nn.LazyLinear(output_size)
            if index < (num_layers - 1):
                self._layers_ordered_dict["relu_" + str(index)] = nn.ReLU()
        
        # Create a sequential container for the layers.
        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    # Define the MLP architecture via output sizes.
    output_sizes = [100, 200, 300]
    
    # Instantiate the LazyMLPBlock model.
    model = LazyMLPBlock(output_sizes)

    print("== Model Architecture ==")
    print(model)
    
    # Create a dummy input tensor.
    # For instance, assume a batch size of 4 and an input feature dimension of 50.
    x = torch.randn(4, 50)
    
    # Print the input tensor shape.
    print("Input shape:", x.shape)
    
    # Pass the input through the MLP.
    output = model(x)
    
    # Print the output tensor shape.
    # The final output should have dimension 300 as specified in the output_sizes.
    print("Output shape:", output.shape)