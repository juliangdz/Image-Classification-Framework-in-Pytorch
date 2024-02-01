import torch
import torchvision.models as models

class TLManager: # Transfer Learning Manager
    def __init__(self, model_name:str,num_classes:int):
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = self._get_pretrained_model()

    def _get_pretrained_model(self):
        model_func = {
            'vit': models.vit_b_16,
            'vgg16': models.vgg16,
            'inception_v3': models.inception_v3,
            'efficientnet_b0': models.efficientnet_b0,
            'googlenet': models.googlenet,
            'resnet18': models.resnet18,
        }

        if self.model_name not in model_func:
            raise ValueError(f"Model {self.model_name} not supported.")

        model = model_func[self.model_name](pretrained=True)

        # Customize the model for the number of classes
        if self.model_name in ['resnet18', 'vgg16', 'googlenet', 'efficientnet_b0']:
            num_ftrs = model.fc.in_features if self.model_name != 'vgg16' else model.classifier[6].in_features
            final_layer = torch.nn.Linear(num_ftrs, self.num_classes)
            if self.model_name == 'vgg16':
                model.classifier[6] = final_layer
            else:
                model.fc = final_layer
        elif self.model_name == 'inception_v3':
            # Handle Inception since its auxiliary output also needs adjustment
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, self.num_classes)
            num_ftrs_aux = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = torch.nn.Linear(num_ftrs_aux, self.num_classes)
        elif self.model_name == 'vit':
            num_ftrs = model.head.in_features
            model.head = torch.nn.Linear(num_ftrs, self.num_classes)
            
        return model

    def get_model(self):
        return self.model