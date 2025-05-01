import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance

class Losses(nn.Module):
    def __init__(self, cfg, device, template_info=None):
        super(Losses, self).__init__()
        self.device = device
        self.template_info = template_info

        self.loss_functions = nn.ModuleDict({
            'vertex_loss': VertexLossL1(),
            'keypoint_loss': KeypointLossL1(self.template_info),
            'chamfer_loss': ChamferLoss()
        })
        
        self.vertex_loss_compare = VertexLossL1()
        self.keypoint_loss_compare =  KeypointLossL1((self.template_info))

        # get pose align loss weight dict
        self.loss_weights = {k: torch.tensor(v, device=self.device) for k, v in cfg.loss_weights.items()}

        self.use_chamfer = cfg.use_chamfer
        self.chamfer_active_epoch = cfg.chamfer_active_epoch

    def forward(self, pred_mesh, true_mesh, epoch):
        """
        Computes weighted sum of losses.
        Returns total loss and a dictionary of individual losses.
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)
        for loss_name, loss_fn in self.loss_functions.items():
                if loss_name == 'chamfer_loss': # check the activated state of chamfer loss
                    if not self.use_chamfer or epoch < self.chamfer_active_epoch:
                        continue
                losses[loss_name] = loss_fn(pred_mesh, true_mesh)
                total_loss += self.loss_weights[loss_name] * losses[loss_name]
                # if loss_name == 'chamfer_loss':
                #     print('I am using chamfer loss.')
                #     print(f"vertex loss is {losses['vertex_loss']}")
                #     print(f"keypoint loss is {losses['keypoint_loss']}")
                #     print(f'chamfer loss is {losses[loss_name]}')
        
        # for comparing differnet loss
        compare_loss = torch.tensor(0.0, device=self.device)
        compare_loss += self.vertex_loss_compare(pred_mesh, true_mesh)
        compare_loss += self.keypoint_loss_compare(pred_mesh, true_mesh)
        return total_loss, losses, compare_loss

class VertexLoss(nn.Module):
    def __init__(self):
        super(VertexLoss, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(beta=1.0)  # Smooth L1 Loss function

    def forward(self, pred_mesh, true_mesh):
        '''
        Calculate mean vertex loss of each batch.
        Input:  pred_mesh: batch_size * num_vertices * 3, 
                true_mesh: batch_size * num_vertices * 3
        Output: Loss_scalar
        '''
        return self.smoothl1(pred_mesh, true_mesh)

class KeypointLoss(nn.Module):
    def __init__(self, template_info):
        super(KeypointLoss, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(beta=1.0)  # Smooth L1 Loss function
        self.keypoint_idx = torch.tensor(template_info['keypoint_idx'])

    def forward(self, pred_mesh, true_mesh):
        '''
        Calculate keypoints loss of each batch. 
        In another word, add more weights to keypoints.
        Input:  pred_mesh: batch_size * num_vertices * 3, 
                true_mesh: batch_size * num_vertices * 3
        Output: Loss_scalar
        '''
        return self.smoothl1(pred_mesh[:, self.keypoint_idx], true_mesh[:, self.keypoint_idx]) 
    
class VertexLossL1(nn.Module):
    def __init__(self):
        super(VertexLossL1, self).__init__()
        self.l1 = nn.L1Loss()  # L1 Loss function

    def forward(self, pred_mesh, true_mesh):
        '''
        Calculate mean vertex loss of each batch.
        Input:  pred_mesh: batch_size * num_vertices * 3, 
                true_mesh: batch_size * num_vertices * 3
        Output: Loss_scalar
        '''
        return self.l1(pred_mesh, true_mesh)

class KeypointLossL1(nn.Module):
    def __init__(self, template_info):
        super(KeypointLossL1, self).__init__()
        self.l1 = nn.L1Loss()  # L1 Loss function
        self.keypoint_idx = torch.tensor(template_info['keypoint_idx'])

    def forward(self, pred_mesh, true_mesh):
        '''
        Calculate keypoints loss of each batch. 
        In another word, add more weights to keypoints.
        Input:  pred_mesh: batch_size * num_vertices * 3, 
                true_mesh: batch_size * num_vertices * 3
        Output: Loss_scalar
        '''
        return self.l1(pred_mesh[:, self.keypoint_idx], true_mesh[:, self.keypoint_idx]) 

class ChamferLoss(nn.Module):
    def __init__(self, unichamfer=True):
        super(ChamferLoss, self).__init__()
        self.unichamfer = unichamfer

    def forward(self, pred_mesh, true_mesh):
        '''
        Compute the Chamfer Distance loss (forward and backward).
        Input:  pred_vertices: batch_size * num_vertices * 3, (B * N * 3)
                true_points: batch_size * num_points * 3 (B * M * 3)
        Output: Chamfer Loss (scalar)
        '''

        # Compute Chamfer Distance step-by-step
        pred_expand = pred_mesh.unsqueeze(2)  # (B, N, 1, 3)
        true_expand = true_mesh.unsqueeze(1)  # (B, 1, M, 3)

        # Compute pairwise distances
        distances = torch.sum((pred_expand - true_expand) ** 2, dim=-1)  # (B, N, M)

        # Compute forward Chamfer Distance
        forward_min_distances, _ = torch.min(distances, dim=2)  # (B, N)
        chamfer_forward = torch.mean(forward_min_distances)  # Scalar

        # Compute backward Chamfer Distance
        backward_min_distances, _ = torch.min(distances, dim=1)  # (B, M)
        chamfer_backward = torch.mean(backward_min_distances)  # Scalar

        if self.unichamfer:
            return chamfer_backward
        else:
            return chamfer_forward + chamfer_backward

# Define the main function to test VertexLoss
def main_vertex_loss():
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define fixed input tensors (Batch_size=4, Num_vertices=3, 3D coordinates)
    pred_mesh = torch.tensor([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],  # Sample 1
        [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]],  # Sample 2
        [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], # Sample 3
        [[2.5, 3.5, 4.5], [5.5, 6.5, 7.5], [8.5, 9.5, 10.5]]  # Sample 4
    ], dtype=torch.float32).to(device)

    true_mesh = torch.tensor([
        [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],  # Sample 1
        [[1.4, 2.4, 3.4], [4.4, 5.4, 6.4], [7.4, 8.4, 9.4]],  # Sample 2
        [[2.1, 3.1, 4.1], [5.1, 6.1, 7.1], [8.1, 9.1, 10.1]], # Sample 3
        [[2.6, 3.6, 4.6], [5.6, 6.6, 7.6], [8.6, 9.6, 10.6]]  # Sample 4
    ], dtype=torch.float32).to(device)

    # Print input values
    print("\nPredicted Mesh:\n", pred_mesh.cpu().numpy())
    print("\nGround Truth Mesh:\n", true_mesh.cpu().numpy())

    # Initialize the VertexLoss function and move it to the correct device
    loss_fn = VertexLoss().to(device)

    # Compute vertex loss
    loss_value = loss_fn(pred_mesh, true_mesh)

    # Step-by-step printout for debugging
    element_wise_loss = nn.SmoothL1Loss(reduction='none', beta=1.0)(pred_mesh, true_mesh)  # Compute element-wise Smooth L1 Loss
    print("\nElement-wise Smooth L1 Loss:\n", element_wise_loss.cpu().numpy())

    mean_loss = element_wise_loss.mean()  # Mean over all elements
    print("\nFinal Mean Smooth L1 Loss (Scalar):", mean_loss.item())

    # Print final loss computed by the model
    print("\nLoss Computed by Model:", loss_value.item())

# Define the main function to test KeypointLoss
def main_keypoint_loss():
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define keypoint indices (e.g., select vertices 0 and 2 as keypoints)
    template_info = {'keypoint_idx': [0, 2]}  # Select first and third vertex as keypoints

    # Define fixed input tensors (Batch_size=2, Num_vertices=3, 3D coordinates)
    pred_mesh = torch.tensor([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],  # Sample 1
        [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]]   # Sample 2
    ], dtype=torch.float32).to(device)

    true_mesh = torch.tensor([
        [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],  # Sample 1
        [[1.4, 2.4, 3.4], [4.4, 5.4, 6.4], [7.4, 8.4, 9.4]]   # Sample 2
    ], dtype=torch.float32).to(device)

    # Print input values
    print("\nPredicted Mesh:\n", pred_mesh.cpu().numpy())
    print("\nGround Truth Mesh:\n", true_mesh.cpu().numpy())

    # Initialize the KeypointLoss function and move it to the correct device
    loss_fn = KeypointLoss(template_info).to(device)

    # Compute keypoint loss
    loss_value = loss_fn(pred_mesh, true_mesh)

    # Step-by-step breakdown
    keypoint_idx = loss_fn.keypoint_idx.to(device)  # Ensure it's on the same device
    pred_keypoints = pred_mesh[:, keypoint_idx]  # Extract keypoint vertices
    true_keypoints = true_mesh[:, keypoint_idx]

    print("\nSelected Keypoint Indices:", keypoint_idx.cpu().numpy())
    print("\nPredicted Keypoints:\n", pred_keypoints.cpu().numpy())
    print("\nGround Truth Keypoints:\n", true_keypoints.cpu().numpy())

    # Compute element-wise Smooth L1 loss
    element_wise_loss = nn.SmoothL1Loss(reduction='none', beta=1.0)(pred_keypoints, true_keypoints)
    print("\nElement-wise Smooth L1 Loss:\n", element_wise_loss.cpu().numpy())

    # Compute mean Smooth L1 loss
    mean_loss = element_wise_loss.mean()
    print("\nFinal Mean Smooth L1 Loss (Scalar):", mean_loss.item())

    # Print final loss computed by the model
    print("\nLoss Computed by Model:", loss_value.item())

# Define the main function to test ChamferLoss
def main_chamfer_loss():
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define fixed input tensors (Batch_size=2, Num_vertices=3, 3D coordinates)
    pred_mesh = torch.tensor([
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],  # Sample 1
    ], dtype=torch.float32).to(device)

    true_mesh = torch.tensor([
        [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]],  # Sample 1
    ], dtype=torch.float32).to(device)

    # Print input values
    print("\nPredicted Mesh:\n", pred_mesh.cpu().numpy())
    print("\nGround Truth Mesh:\n", true_mesh.cpu().numpy())

    # Initialize ChamferLoss function and move it to the correct device
    loss_fn = ChamferLoss(unichamfer=True).to(device)

    # Compute Chamfer loss
    loss = loss_fn(pred_mesh, true_mesh)

    # Print final loss computed by the model
    print("\nLoss Computed by Model:", loss)

# Run the main function
if __name__ == "__main__":
    main_chamfer_loss()