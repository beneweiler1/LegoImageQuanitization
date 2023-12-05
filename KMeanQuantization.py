import random
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

filepath = "./masterSword.png"
colors = [(249,108,98),
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

# Iterate over each section
for sx in range(num_sections_x):
    for sy in range(num_sections_y):
        left = sx * section_width
        upper = sy * section_height
        right = left + section_width
        lower = upper + section_height

        # Crop the section from the original image
        section = input_image.crop((left, upper, right, lower))

        # Convert the section to a NumPy array
        section_array = np.array(section)

        # Reshape the section array to a flat list of pixels
        pixels = section_array.reshape(-1, 3)

        # Apply K-means clustering to find representative colors
        n_clusters = 1  # You can adjust this value to control color representation
        kmeans = KMeans(n_clusters=1, random_state=0, n_init=10).fit(pixels)

        cluster_centers = kmeans.cluster_centers_

        # Create a solid color section using the mean color of the cluster centers
        mean_color = tuple(map(int, cluster_centers.mean(axis=0)))
        solid_section = Image.new("RGB", section.size, mean_color)

        # Paste the solid section into the output image
        output_image.paste(solid_section, (left, upper))


# Save or display the reconstructed image
output_image.show()
