import torch
import pickle
import numpy as np
import torch.nn as nn

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model import gat_model_all, resnet_model

class ClothModel(nn.Module):
    """
        Template-based Mass-spring Cloth GNN
    """
    def __init__(self, template_info, message_passing_steps=15):
        super(ClothModel, self).__init__()

        # init template_info
        self.template_info = template_info
        self.template_mesh_pos = self.template_info['mesh_pos']
        self.template_edge_idx = self.template_info['edge_idx'].astype(int)

        # init backbone ResNet model
        self.backbone_model = resnet_model.get_model('resnet34')

        # init learned GAT model
        self.learned_model = gat_model.GATModel(
            output_size=3,
            latent_size=128,
            num_layers=2,
            message_passing_steps=message_passing_steps
        )

    # build template_graph with template_nodes, edge_senders, and edge_receivers
    def _build_template_graph(self):
        # init node features as template mesh positions: N (num of nodes) x 3 (positions)
        node_features = torch.from_numpy(self.template_mesh_pos).float().cuda()
      
        # create two-way connectivity of edge senders and receivers
        senders = self.template_edge_idx[:, 0]
        receivers = self.template_edge_idx[:, 1]
        sender = torch.from_numpy(np.concatenate([senders, receivers], 0)).to(torch.int64).cuda()
        receiver = torch.from_numpy(np.concatenate([receivers, senders], 0)).to(torch.int64).cuda()
        
        # assign edge features as edge vertices' relative coordinate and norm: E (num of edges) x 4 (vecter + norm)
        relative_edge_vector = (torch.index_select(node_features, 0, sender) - torch.index_select(node_features, 0, receiver))
        relative_edge_norm = torch.norm(relative_edge_vector, dim=-1, keepdim=True)
        edge_features = torch.cat((relative_edge_vector, relative_edge_norm), -1)

        # return gat model with node features and mesh edges
        mesh_edges = gat_model.EdgeSet(
            name='mesh_edges',
            features=edge_features,
            receivers=receiver,
            senders=sender
        )

        return gat_model.GraphSet(node_features=node_features, edge_set=mesh_edges)

    def forward(self, batch):
        
        # resnet encode image feature
        image_feature = self.backbone_model(batch)
        
        # build template graph from template_info
        template_graph = self._build_template_graph()
        
        # forward gat with message_passing_steps
        pred_mesh = self.learned_model(template_graph, image_feature)

        # batch_size, vertex_num, vertex_dim(3: xyz position)
        b, n_v, n_p = pred_mesh.shape
        
        return pred_mesh

# ------------------------------------------------------ #
# ------------------- ResNet Encoder ------------------- #
# ------------------------------------------------------ #

class ResNet(nn.Module):
    """
    This is a resnet wrapper class which takes existing resnet architectures and
    adds a final linear layer at the end, ensuring proper output dimensionality
    """
    def __init__(self, model_name, pretrained):
        super().__init__()
        
        if pretrained:
            if model_name == "resnet18":
                model_func = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            elif model_name == "resnet34":
                model_func = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            elif model_name == "resnet50":
                model_func = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            elif model_name == "resnet101":
                model_func = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            elif model_name == "resnet152":
                model_func = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
            else:
                raise Exception(f"Unknown backend model type: {model_name}")
        else:
            if model_name == "resnet18":
                model_func = models.resnet18
            elif model_name == "resnet34":
                model_func = models.resnet34
            elif model_name == "resnet50":
                model_func = models.resnet50
            elif model_name == "resnet101":
                model_func = models.resnet101
            elif model_name == "resnet152":
                model_func = models.resnet152
            else:
                raise Exception(f"Unknown backend model type: {model_name}")

        # construct the resnet encoder
        if pretrained:
            b_model = model_func
        else:
            b_model = model_func()
        resnet_encoder = nn.Sequential(
            b_model.conv1,
            b_model.bn1,
            b_model.relu,
            b_model.maxpool,
            b_model.layer1,
            b_model.layer2,
            b_model.layer3,
            b_model.layer4,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.encoder = resnet_encoder

    def forward(self, x):
        # batch size, channels, height, width
        b, c, h, w = x.shape
        
        # get feature output
        f = self.encoder(x)
        
        # get final output
        f = f.flatten(start_dim=1)

        # batch_size, feature_dim (512)
        b, l = f.shape

        return f

# ------------------------------------------------------ #
# ---------------- Access ResNet Model ----------------- #
# ------------------------------------------------------ #

def get_model(model_name, pretrained=False):
    model = ResNet(model_name=model_name, pretrained=pretrained)
    return model

# test the model
if __name__ == "__main__":
    model = get_model("resnet18", pretrained=True)
    print(model)