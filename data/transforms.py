from albumentations.pytorch import ToTensorV2
import albumentations as A

def apply_transform(transform_config: list):
    transform_list = []
    transform_mapping = {
        "Resize": lambda params: A.Resize(**params),
        "CenterCrop": lambda params: A.CenterCrop(**params),
        "Normalize": lambda params: A.Normalize(**params),
        "ToTensor": lambda params: ToTensorV2()
    }

    for transform in transform_config:
        transform_func = transform_mapping.get(transform["name"])
        if transform_func:
            transform_list.append(transform_func(transform.get("params", {})))
        else:
            raise ValueError(f"Transform {transform['name']} not recognized")
    
    composed_transform = A.Compose(transform_list)
    return composed_transform
