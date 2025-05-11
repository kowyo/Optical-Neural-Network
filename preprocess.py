import torch

def preprocess_upscale_pad(inputs):
    """
    Common preprocessing for both teacher and ONN models:
    - Upscale from 28x28 to 150x150
    - Pad from 150x150 to 200x200
    """
    # Upscale the images from 28x28 to 150x150
    upscaled = torch.nn.functional.interpolate(inputs, size=(150, 150), mode='bilinear', align_corners=False)
    
    # Pad the images from 150x150 to 200x200
    padded = torch.nn.functional.pad(upscaled, (25, 25, 25, 25))
    
    return padded

def preprocess_for_onn(inputs):
    """
    Convert standard Fashion-MNIST images to the format expected by ONN:
    - Upscale from 28x28 to 150x150
    - Pad from 150x150 to 200x200
    - Add complex dimension (real and imaginary parts)
    """
    # Apply common preprocessing
    padded = preprocess_upscale_pad(inputs)
    
    # Convert from [batch, 1, 200, 200] to [batch, 200, 200]
    padded = padded.squeeze(1)
    
    # Add complex dimension (real part is the image, imaginary part is zeros)
    # Shape becomes [batch, 200, 200, 2]
    complex_input = torch.stack((padded, torch.zeros_like(padded)), dim=-1)
    
    return complex_input