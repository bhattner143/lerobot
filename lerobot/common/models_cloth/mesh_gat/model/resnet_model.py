import torch.nn as nn
import torchvision.models as models


# resnet model in the torchvision library
def resnet_list():
    return ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

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