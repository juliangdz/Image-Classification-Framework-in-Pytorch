import torch
from torchviz import make_dot
import tempfile
import os
from PIL import Image

def generate_model_graph(model, input_tensor):
    # Generate a forward pass to get the model output
    output = model(input_tensor)
    
    # Create a dot graph of the model
    dot = make_dot(output, params=dict(model.named_parameters()))
    
    return dot

def dot_to_image(dot):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        dot.render(tmpfile.name, format='png')
        tmpfile.close()  # Close the file so we can read it later
        image = Image.open(tmpfile.name)  # Open the image using PIL
        os.unlink(tmpfile.name)  # Delete the temp file
    return image