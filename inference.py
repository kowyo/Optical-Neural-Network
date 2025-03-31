import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import onn

def main(args):
    # Setup the device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    
    # Load test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = torchvision.datasets.FashionMNIST(
        "./data", train=False, transform=transform, download=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
    )
    
    # Load model
    model = onn.Net()
    model_path = os.path.join(args.model_path, args.model_name)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f'Model loaded from "{model_path}"')
    else:
        print(f"Model file not found at {model_path}")
        return
    
    model.to(device)
    model.eval()
    
    # Run inference
    all_preds = []
    all_labels = []
    all_images = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Running inference"):
            # Save original images for visualization
            if len(all_images) < args.num_samples:
                all_images.extend(images[:args.num_samples - len(all_images)])
            
            # Preprocess images
            images = images.to(device)
            labels = labels.to(device)
            
            # Apply padding as in training
            images_padded = F.pad(images, pad=(86, 86, 86, 86))
            
            # Format for model input (adding complex part)
            images_complex = torch.squeeze(
                torch.cat(
                    (images_padded.unsqueeze(-1), torch.zeros_like(images_padded.unsqueeze(-1))),
                    dim=-1
                ),
                dim=1
            )
            
            # Forward pass
            outputs = model(images_complex)
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if len(all_preds) >= args.num_samples and args.num_samples > 0:
                break
    
    # Calculate accuracy
    if len(all_preds) > 0:
        accuracy = np.mean(np.array(all_preds[:len(all_labels)]) == np.array(all_labels)) * 100
        print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Visualize results
    if args.visualize and all_images:
        visualize_results(all_images[:args.num_samples], 
                         all_preds[:args.num_samples], 
                         all_labels[:args.num_samples],
                         args.save_path)
        
def visualize_results(images, predictions, ground_truth, save_path=None):
    """Visualize the model predictions compared to ground truth."""
    num_samples = min(len(images), 25)  # Show at most 25 images
    rows = int(np.ceil(num_samples / 5))
    cols = min(num_samples, 5)
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2*rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = images[i].squeeze().cpu().numpy()
        pred = predictions[i]
        gt = ground_truth[i]
        
        axes[i].imshow(img, cmap='gray')
        color = 'green' if pred == gt else 'red'
        axes[i].set_title(f'Pred: {pred} (GT: {gt})', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Model Predictions on Test Data', y=1.02, fontsize=16)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./saved_model/")
    parser.add_argument("--model-name", type=str, default="400_model.pth")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=25, 
                        help="Number of samples to visualize. Set to 0 to evaluate the entire dataset without visualization")
    parser.add_argument("--visualize", type=bool, default=True)
    parser.add_argument("--save-path", type=str, default="./results/inference_results.png")
    
    args = parser.parse_args()
    main(args)
