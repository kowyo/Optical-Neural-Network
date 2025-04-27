import matplotlib.pyplot as plt
import re
import os

def parse_log_file(filepath):
    """Parses the log file to extract epoch and accuracy."""
    epochs = []
    accuracies = []
    # Regex to find lines with Epoch and Accuracy
    pattern = re.compile(r"Epoch (\d+)/\d+, Loss: [\d.]+, Accuracy: ([\d.]+)%")
    try:
        with open(filepath, 'r') as f:
            for line in f:
                match = pattern.match(line)
                if match:
                    epochs.append(int(match.group(1)))
                    accuracies.append(float(match.group(2)))
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None
    return epochs, accuracies

# File paths (assuming the script is run from the Optical-Neural-Network directory)
with_kd_file = 'with_kd.txt'
without_kd_file = 'without_kd.txt'

# Parse the data
epochs_with_kd, acc_with_kd = parse_log_file(with_kd_file)
epochs_without_kd, acc_without_kd = parse_log_file(without_kd_file)

# Check if parsing was successful
if acc_with_kd is None or acc_without_kd is None:
    print("Exiting due to file parsing errors.")
else:
    # Plotting
    plt.figure(figsize=(10, 6))

    plt.plot(epochs_with_kd, acc_with_kd, label='With Knowledge Distillation')
    plt.plot(epochs_without_kd, acc_without_kd, label='Without Knowledge Distillation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=min(min(acc_with_kd), min(acc_without_kd)) - 1, top=max(max(acc_with_kd), max(acc_without_kd)) + 1) # Adjust y-axis limits slightly
    plt.tight_layout()

    # Save the plot
    plot_filename = 'accuracy_comparison.png'
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")

    # Show the plot (optional, comment out if running in a non-GUI environment)
    # plt.show()