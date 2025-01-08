import torch
import os

def export_model_to_onnx(model, dummy_input, export_path, export_name="model.onnx"):
    model.eval()
    os.makedirs(export_path, exist_ok=True)
    onnx_file = os.path.join(export_path, export_name)

    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_file,
        export_params=True,
        opset_version=11,    # check if your opset version is suitable
        do_constant_folding=True,
        input_names = ['input'], 
        output_names = ['output']
    )
    print(f"Model exported to {onnx_file}")
