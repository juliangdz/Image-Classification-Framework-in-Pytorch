import argparse
import torch
import onnxruntime
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

def load_pth_model(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def load_onnx_model(onnx_path):
    # Create an onnxruntime session
    session = onnxruntime.InferenceSession(onnx_path)
    return session

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    augmented = transform(image=image)
    tensor_img = augmented['image']  # Albumentations output
    if not isinstance(tensor_img, torch.Tensor):
        tensor_img = ToTensorV2()(image=augmented['image'])['image']
    return tensor_img

def infer_pth(model, image_tensor, device):
    image_tensor = image_tensor.unsqueeze(0).to(device)  # add batch dim
    with torch.no_grad():
        outputs = model(image_tensor)
        preds = torch.argmax(outputs, dim=1)
    return preds.cpu().item()

def infer_onnx(session, image_tensor):
    # Convert the image_tensor to numpy
    image_numpy = image_tensor.unsqueeze(0).cpu().numpy()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_numpy})
    # outputs is a list, the first element is your model's output
    preds = np.argmax(outputs[0], axis=1)
    return preds[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['pth','onnx'],
                        help='Type of model to load.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the .pth or .onnx file')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the image for inference')
    parser.add_argument('--config', type=str, required=False, default='config.json',
                        help='Path to the config.json (for transformations, etc.)')
    args = parser.parse_args()

    # Read config for transformations
    import json
    with open(args.config,'r') as f:
        config = json.load(f)

    transform_config = config['data']['transforms']['options'][config['data']['transforms']['name']]
    # Build albumentations transform
    from data.transforms import apply_transform
    transform = apply_transform(transform_config)

    # Preprocess image
    image_tensor = preprocess_image(args.image_path, transform)

    if args.model_type == 'pth':
        # We must build the same architecture as training
        from networks.runner import build_model
        from utils.helper import get_device
        device = get_device()
        model = build_model(config['hyperparams']['network'], config)
        model.to(device)
        model = load_pth_model(model, args.model_path, device)
        pred = infer_pth(model, image_tensor, device)
        print(f"Prediction using PyTorch .pth model: {pred}")
    else:
        session = load_onnx_model(args.model_path)
        pred = infer_onnx(session, image_tensor)
        print(f"Prediction using ONNX model: {pred}")
