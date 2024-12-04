import random
from PIL import Image, ImageDraw
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance

filepath = "../assets/poke007.png"
target_colors = [(249,108,98),
           (245, 125, 32),
           (251,171,24), 
           (252,195,158),
           (227,224,41),
           (0,175,77),
           (24,158,159),
           (132,200,226),
            (0,57,94),
            (255,255,255)]

input_image = Image.open(filepath)
# Define the number of sections in both dimensions
# Define the number of sections in both dimensions
num_sections_x = 80
num_sections_y = 80

# Calculate the section size based on the dimensions of the input image
section_width = input_image.width // num_sections_x
section_height = input_image.height // num_sections_y

# Create a blank canvas to reconstruct the image
output_image = Image.new("RGB", input_image.size)

inset_factor = 0.1

section_colors = []

for sx in range(num_sections_x):
    for sy in range(num_sections_y):
        left = sx * section_width
        upper = sy * section_height
        right = left + section_width
        lower = upper + section_height

        # Crop the section from the original image
        section = input_image.crop((left, upper, right, lower))

        # Calculate the average color of the section
        section_array = np.array(section)
        average_color = tuple(map(int, np.mean(section_array, axis=(0, 1))))

        # Add the average color to the list of section colors
        section_colors.append(average_color)

# Apply K-means clustering to the section colors
n_clusters = len(target_colors)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(section_colors)

# Get the cluster centroids (representative colors)
cluster_centers = kmeans.cluster_centers_.astype(int)

# Map the cluster centroids to the target colors
color_mapping = {}
for i in range(n_clusters):
    closest_target_color = min(target_colors, key=lambda x: np.linalg.norm(np.array(x) - cluster_centers[i]))
    color_mapping[tuple(cluster_centers[i])] = closest_target_color

# Create a drawing context for the output image
draw = ImageDraw.Draw(output_image)

# Iterate over each section again
section_index = 0
for sx in range(num_sections_x):
    for sy in range(num_sections_y):
        left = sx * section_width
        upper = sy * section_height
        right = left + section_width
        lower = upper + section_height

        # Get the cluster label for the section's color
        cluster_label = kmeans.labels_[section_index]

        # Get the corresponding target color from the mapping
        target_color = color_mapping[tuple(cluster_centers[cluster_label])]

        # Create a solid color section with the target color
        solid_section = Image.new("RGB", (section_width, section_height), target_color)

        # Create a mask for the circular shape
        mask = Image.new("L", (section_width, section_height), 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.ellipse((0, 0, section_width, section_height), fill=255)

        # Paste the solid color section into the output image using the circular mask
        output_image.paste(solid_section, (left, upper), mask)

        section_index += 1


# Save or display the reconstructed image
output_image.show()
