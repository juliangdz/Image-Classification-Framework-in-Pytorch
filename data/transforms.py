from torchvision import transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A


def apply_transform(transform_config:dict):
    # List to hold transforms
    transform_list = []

    # Mapping from config names to actual transformation functions
    transform_mapping = {
        "Resize": lambda params: A.Resize(**params),
        "Normalize": lambda params: A.Normalize(**params),
        "ToTensor": lambda params: ToTensorV2()
    }

    for transform in transform_config:
        transform_func = transform_mapping.get(transform["name"])
        if transform_func:
            # Handle special case for ToTensor
            if transform["name"] == "ToTensor":
                transform_list.append(transform_func({}))
            else:
                transform_list.append(transform_func(transform.get("params", {})))
        else:
            raise ValueError(f"Transform {transform['name']} not recognized")
        
    # Compose all transforms into one
    composed_transform = A.Compose(transform_list)

    return composed_transform