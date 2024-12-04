from PIL import Image
import numpy as np
import pandas as pd

# Load your input image
filepath = "../../assets/poke007.png"
input_image = Image.open(filepath).convert("RGB")

# Resize the input image to 58x58
pixelated_image = input_image.resize((58, 58), Image.Resampling.NEAREST)

# Helper function to convert HEX to RGB
def hex_to_rgb(hex_code):
    try:
        hex_code = hex_code.lstrip('#')
        return tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))
    except (ValueError, TypeError):
        return None

# Load the CSV file
data = pd.read_csv('../../assets/perlercolor - FULL.csv')

# Extract the HTML color codes and convert them to RGB
target_colors = [hex_to_rgb(color) for color in data['HTML'].dropna() if hex_to_rgb(color) is not None]

# Ensure all target colors are valid RGB tuples
target_colors = np.array([color for color in target_colors if len(color) == 3])

# Convert the pixelated image to a numpy array
pixelated_pixels = np.array(pixelated_image)

# Reshape the pixels for easier distance computation
reshaped_pixels = pixelated_pixels.reshape((-1, 3))

# Compute distances from each pixel to each target color
distances = np.linalg.norm(reshaped_pixels[:, None, :] - target_colors[None, :, :], axis=2)

# Find the index of the closest target color for each pixel
closest_color_indices = np.argmin(distances, axis=1)

# Map each pixel to its closest target color
output_pixels = target_colors[closest_color_indices]

# Reshape the output back to 58x58
output_pixels = output_pixels.reshape((58, 58, 3))

# Add white borders directly into the 58x58 grid
bordered_pixels = np.full((58, 58, 3), (255, 255, 255), dtype=np.uint8)  # Start with a white image
for x in range(58):
    for y in range(58):
        if x % 2 == 1 and y % 2 == 1:  # Only color every alternate position to simulate borders
            bordered_pixels[x, y] = output_pixels[x // 2, y // 2]

# Create the output image
output_image = Image.fromarray(bordered_pixels.astype('uint8'))

# Save or display the reconstructed image
output_image.save("pixelated_output_true_58x58.jpg")
output_image.show()
