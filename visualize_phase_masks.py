import torch
import numpy as np
import matplotlib.pyplot as plt
from onn import Net
import os

def visualize_phase_masks(model_path):
    """
    Load a trained ONN model and visualize the phase masks of each layer
    
    Args:
        model_path: Path to the saved model file
    """
    # Load the model
    try:
        model = Net(num_layers=5)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Extract phase masks from the model
    phase_masks = []
    for i in range(5):  # Assuming 5 layers as in the original model
        # Get the parameter named "phase_0", "phase_1", etc.
        phase_param = getattr(model, f"phase_{i}")
        # Ensure phase values are between 0 and π
        phase_mask = phase_param.detach().cpu().numpy() % (np.pi)
        phase_masks.append(phase_mask)
    
    # Create a figure to display all phase masks
    fig, axes = plt.subplots(1, len(phase_masks), figsize=(20, 5))
    
    for i, phase_mask in enumerate(phase_masks):
        im = axes[i].imshow(phase_mask, cmap='jet', vmin=0, vmax=np.pi)
        axes[i].set_title(f"Layer {i+1} Phase Mask")
        axes[i].axis('off')
    
    # Add a colorbar to show the phase values
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', 
                        fraction=0.046, pad=0.04, aspect=30)
    cbar.set_label('Phase (radians)')
    cbar.set_ticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    cbar.set_ticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])
    
    plt.tight_layout()
    plt.savefig('phase_masks.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Path to the trained model - use the same path as in visualize_onn.py
    model_path = 'onn_student/onn_student_epoch_10.pth'
    visualize_phase_masks(model_path)
    
    # Alternatively, visualize multiple epochs to see phase mask evolution
    """
    # Uncomment this section to visualize phase evolution across epochs
    for epoch in range(1, 11):
        model_path = f'onn_student/onn_student_epoch_{epoch}.pth'
        plt.figure(figsize=(8, 8))
        plt.suptitle(f"Phase Masks Evolution - Epoch {epoch}")
        visualize_phase_masks(model_path)
        plt.savefig(f'phase_masks_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    """
