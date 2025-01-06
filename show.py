from PIL import Image
import numpy as np

def semantic_to_color(input_path, output_path, colormap):
    """
    Converts a semantic segmentation image to a colorized version based on a colormap.

    Args:
        input_path (str): Path to the input semantic segmentation PNG file.
        output_path (str): Path to save the colorized PNG file.
        colormap (dict): A dictionary mapping class indices to RGB colors.
    """
    # Load the semantic segmentation image
    semantic_img = np.array(Image.open(input_path))

    # Handle RGBA or multi-channel input by extracting one channel
    if semantic_img.ndim == 3 and semantic_img.shape[-1] == 4:
        # Extract the first channel (assuming category information is in the first channel)
        semantic_img = semantic_img[:, :, 0]
    elif semantic_img.ndim == 3 and semantic_img.shape[-1] == 3:
        # Convert RGB to grayscale (if needed)
        semantic_img = Image.open(input_path).convert("L")
        semantic_img = np.array(semantic_img)

    # Ensure the result is 2D
    if semantic_img.ndim != 2:
        raise ValueError("Input semantic image should be a 2D grayscale image after conversion")

    # Create an empty color image
    color_img = np.zeros((*semantic_img.shape, 3), dtype=np.uint8)

    # Map each class index to the corresponding RGB color
    for class_idx, color in colormap.items():
        color_img[semantic_img == class_idx] = color

    # Save the colorized image
    Image.fromarray(color_img).save(output_path)
    print(f"Colorized image saved to: {output_path}")

# Example colormap for 25 classes
num_classes = 25
colormap = {i: tuple(np.random.randint(0, 256, size=3)) for i in range(num_classes)}

# File paths
input_path = "/home/yi/Documents/DELIVER/data/DELIVER/semantic/cloud/test/MAP_10_point102/045050_semantic_front.png"  # Input semantic PNG file
output_path = "/home/yi/Documents/DELIVER/data/DELIVER/colorized_segmentation.png"  # Output colorized PNG file

# Convert and save
semantic_to_color(input_path, output_path, colormap)


