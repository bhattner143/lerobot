from lerobot.common.policies.act_mesh_gat.configuration_act_mesh_gat import ACTMeshGATConfig
from lerobot.common.models_cloth.clothmodel_configs import*

from lerobot.common.models_cloth.clothmodel import PreTrainedClothModel
import logging
from termcolor import colored

####IMPORT CLOTHMODEL BASED ON NAME #####
def get_cloth_model_class(name: str) -> PreTrainedClothModel:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name == "mesh_gat":
        from lerobot.common.models_cloth.clothmodel import ClothMeshGATModel

        return ClothMeshGATModel
    
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")
    
###### MAKE CLOTH MODEL FROM CLOTH MODEL CONFIG ##########
def make_cloth_model(
        config: PreTrainedClothModelConfig,      
        ) -> PreTrainedClothModel:
    """Create a cloth model instance based on the provided configuration."""
    # Get the class of the cloth model based on the name in the configuration
    cloth_model_class = get_cloth_model_class(config.type)
    # Create an instance of the cloth model class depending on the mode (train or eval)
    if config.mode == 'train':
        kwargs = {
            "config": config
        }
        # TODO: cloth model training to be implemented through the init method of the ClothMeshGATModel class
        cloth_model = cloth_model_class(**kwargs)
    elif config.mode == 'eval':
        pretrained_config_path = config.path_config.config_dir
        kwargs = {
            "pretrained_path": pretrained_config_path,
            "config": config
        }
        cloth_model = cloth_model_class.from_pretrained_cloth_model(
                    **kwargs
                    )
        logging.info(colored(f"Loaded pre-trained cloth model config: {config.type}", "yellow", attrs=["bold"]))
    else:
        raise NotImplementedError(f"Mode {config.mode} is not applicable. Choose between train or eval.")

    return cloth_model