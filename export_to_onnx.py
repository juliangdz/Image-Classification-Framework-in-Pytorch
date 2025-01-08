import argparse
import torch
import pdb
import os
from utils.helper import read_config, get_device
from networks.runner import build_model  # A new function we might create for building the model
from utils.export_onnx import export_model_to_onnx

def main(config_path, checkpoint_path, export_name):
    config = read_config(config_path)
    device = get_device()

    # Build model (similar logic in runner)
    network_name = config['hyperparams']['network']
    
    model = build_model(network_name, config,input_shape=config['data']['input_shape'],num_classes=config['data']['num_classes'])
    model = model.to(device)

    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    input_shape = config["data"]["input_shape"]
    input_shape = [input_shape[-1]] + input_shape[:-1]
    input_shape.insert(0,1)
    
    input_shape = tuple(input_shape)
    print('Input shape : ',input_shape)
    dummy_input = torch.randn(input_shape).to(device)

    export_path = config['onnx']['dir']
    export_model_to_onnx(model, dummy_input, export_path, export_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.json')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--export_name', type=str, default="model.onnx", help='Name of the onnx file')
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.export_name)
