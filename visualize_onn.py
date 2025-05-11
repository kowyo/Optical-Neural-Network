import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import datasets, transforms
from onn import Net, DiffractiveLayer, detector_region

# Function to visualize detector regions
def draw_detector_regions(detector_plane, size=200):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(detector_plane)
    
    # Define detector regions based on onn.py's detector_region function
    regions = [
        (46, 66, 46, 66),   # region 0: (y_start, y_end, x_start, x_end)
        (46, 66, 93, 113),  # region 1
        (46, 66, 140, 160), # region 2
        (85, 105, 46, 66),  # region 3
        (85, 105, 78, 98),  # region 4
        (85, 105, 109, 129),# region 5
        (85, 105, 140, 160),# region 6
        (125, 145, 46, 66), # region 7
        (125, 145, 93, 113),# region 8
        (125, 145, 140, 160)# region 9
    ]
    
    for i, (y_start, y_end, x_start, x_end) in enumerate(regions):
        width = x_end - x_start
        height = y_end - y_start
        rect = patches.Rectangle((x_start, y_start), width, height, 
                                linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_start + width//2, y_start + height//2, str(i), 
                color='white', fontsize=12, ha='center', va='center')
    
    plt.title("Detector Plane with Detection Regions")
    plt.tight_layout()
    return fig

# Function to propagate through network and collect visualizations
def visualize_propagation(model, input_image):
    device = next(model.parameters()).device
    
    # Prepare input (batch_size=1, H=200, W=200)
    if isinstance(input_image, np.ndarray):
        # Convert to tensor if numpy array
        input_tensor = torch.from_numpy(input_image).float().to(device)
    else:
        input_tensor = input_image.float().to(device)
    
    # Ensure proper shape and add complex dimension (real and imaginary parts)
    if len(input_tensor.shape) == 2:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Create complex input (real part = image, imaginary part = 0)
    x = torch.stack((input_tensor, torch.zeros_like(input_tensor)), dim=-1)
    
    # Ensure input image is properly squeezed for visualization
    result_images = [input_tensor.detach().squeeze().cpu().numpy()]
    
    # Propagate through each layer
    for index, layer in enumerate(model.diffractive_layers):
        temp = layer(x)
        exp_j_phase = torch.stack((torch.cos(model.phase[index]), torch.sin(model.phase[index])), dim=-1)
        x_real = temp[..., 0] * exp_j_phase[..., 0] - temp[..., 1] * exp_j_phase[..., 1]
        x_imag = temp[..., 0] * exp_j_phase[..., 1] + temp[..., 1] * exp_j_phase[..., 0]
        x = torch.stack((x_real, x_imag), dim=-1)
        
        # Calculate magnitude for visualization
        x_abs = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        result_images.append(x_abs.detach().squeeze().cpu().numpy())
    
    # Final propagation
    final = model.last_diffractive_layer(x)
    final_abs = torch.sqrt(final[..., 0] ** 2 + final[..., 1] ** 2)
    result_images.append(final_abs.detach().squeeze().cpu().numpy())
    
    # Detector output - fix: use the detector_region function from the onn module
    detector_output = model.softmax(detector_region(final_abs))
    
    return result_images, detector_output, final_abs.detach().squeeze().cpu().numpy()

# Function to load and preprocess a Fashion MNIST image
def load_fashion_mnist_sample(idx):
    # Use the same preprocessing as in the notebook
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Added normalization to match training
        # Pad instead of resize to preserve the image shape
        transforms.Lambda(lambda x: torch.nn.functional.pad(x, (86, 86, 86, 86)))
    ])
    
    # Load Fashion MNIST dataset
    test_dataset = datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Get a sample image
    image, label = test_dataset[idx]
    return image.squeeze().numpy(), label, test_dataset.classes[label]

# Main execution
if __name__ == "__main__":
    # Load a pretrained model or create a new one
    model = Net(num_layers=5)
    
    # Try to load saved model parameters if available
    try:
        # Use the correct path to the model file and map tensors to CPU
        # model_path = 'onn_model/epoch_15.pth'
        model_path = 'onn_student/onn_student_epoch_10.pth'
        # Add map_location to ensure proper device handling
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded pretrained model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using randomly initialized model")
    
    model.eval()
    
    # Load a sample ankle boot image from Fashion MNIST
    # Find an ankle boot (class 9) by searching through the dataset
    test_dataset = datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True
    )
    
    # Find sample indices that correspond to ankle boots
    ankle_boot_indices = [i for i, (_, label) in enumerate(test_dataset) if label == 9]
    
    # Select the second ankle boot (or a different one if you've already used index 0)
    selected_idx = ankle_boot_indices[2]  # Use a different ankle boot
    
    # Load the selected ankle boot
    image, label, class_name = load_fashion_mnist_sample(selected_idx)
    print(f"Sample image index: {label}, class name: {class_name}")
    
    # Visualize propagation through the network
    result_images, detector_output, detector_plane = visualize_propagation(model, image)
    
    # Plot the propagation results
    fig, axes = plt.subplots(1, len(result_images), figsize=(20, 5))
    for i, img in enumerate(result_images):
        # Make sure img is properly squeezed before visualization
        if len(img.shape) > 2:
            img = img.squeeze()  # Remove any extra dimensions
        im = axes[i].imshow(img, cmap='viridis')
        axes[i].set_title(f"Layer {i-1}" if i > 0 else "Input")
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('propagation.png', dpi=300, bbox_inches='tight')
    plt.figure()
    
    # Draw detector regions
    detector_fig = draw_detector_regions(detector_plane)
    plt.savefig('detector_regions.png', dpi=300, bbox_inches='tight')
    
    # Plot detector outputs (classification results)
    plt.figure(figsize=(10, 5))
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    outputs = detector_output.detach().squeeze().cpu().numpy()
    plt.bar(np.arange(10), outputs)
    plt.xticks(np.arange(10), classes, rotation=45)
    plt.title(f"Classification Results (True label: {class_name})")
    plt.tight_layout()
    plt.savefig('classification_results.png', dpi=300, bbox_inches='tight')
    
    print(f"True class: {class_name}")
    print(f"Predicted probabilities: {outputs}")
    print(f"Predicted class: {classes[np.argmax(outputs)]}")
    
    plt.show()
