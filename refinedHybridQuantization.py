from PIL import Image, ImageDraw
import numpy as np
from skimage.feature import canny
from sklearn.cluster import KMeans
from scipy.spatial import distance
# Load your input image
filepath = "./masterSword.png"
input_image = Image.open(filepath)

# Define the number of sections in both dimensions
num_sections_x = 80
num_sections_y = 80

# Calculate the section size based on the dimensions of the input image
section_width = input_image.width // num_sections_x
section_height = input_image.height // num_sections_y

# Define the inset factor
inset_factor = 0.75

# Create a blank canvas to reconstruct the image
output_image = Image.new("RGB", input_image.size)

# Define the list of target colors including the border colors
target_colors = [(249, 108, 98),
                 (245, 125, 32),
                 (251, 171, 24),
                 (252, 195, 158),
                 (227, 224, 41),
                 (0, 175, 77),
                 (24, 158, 159),
                 (132, 200, 226),
                 (0, 57, 94),
                 (255, 255, 255)]  # Border color is (255, 255, 255)

# Create a list to store all section colors
section_colors = []

def colorDetector(color):
    #change this to find the x most common colors and assign them a Random? color from here? 
    colors = [[249,108,98], [245, 125, 32],[251,171,24], [252,195,158],[227,224,41],[0,175,77], [24,158,159], [132,200,226], [0,57,94], [255,255,255]]
    color = np.array(color)
    distances = np.linalg.norm(colors - color, axis=1)
    index_of_smallest = np.argmin(distances)

    return tuple(colors[index_of_smallest])

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
color_mapping = {tuple(cluster_centers[i]): target_colors[i] for i in range(n_clusters)}

# Apply Canny edge detection to the input image
edge_image = canny(np.array(input_image.convert("L")), sigma=1.0)

# Create a circular mask
circle_mask = Image.new("L", (section_width, section_height), 0)
draw_mask = ImageDraw.Draw(circle_mask)
draw_mask.ellipse((0, 0, section_width * inset_factor, section_height * inset_factor), fill=255)

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

        # Get the edge mask for the current section
        section_edge = edge_image[upper:lower, left:right]

        # Determine if this section contains an edge
        has_edge = np.any(section_edge)


        section = input_image.crop((left, upper, right, lower))

        # Calculate the average color of the section
        section_array = np.array(section)
        average_color = tuple(map(int, np.mean(section_array, axis=(0, 1))))
    
        # If there's an edge, use one of the edge target colors from target_colors
        if has_edge:
            output_image.paste(colorDetector(average_color), (left, upper), circle_mask)
            #edge_color_index = n_clusters - 4  # Start from the first edge color
            #output_color = target_colors[edge_color_index]
            #
        else:
           output_image.paste(solid_section, (left, upper), circle_mask)

        # # Create a solid color section with the output color
        # output_section = Image.new("RGB", (section_width, section_height), output_color)

        # # Paste the section into the output image
        # output_image.paste(output_section, (left, upper), circle_mask)

        section_index += 1

output_image.show()
