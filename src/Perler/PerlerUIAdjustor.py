import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import pandas as pd
from collections import Counter

# Helper Functions
def hex_to_rgb(hex_code):
    try:
        hex_code = hex_code.lstrip('#')
        return tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))
    except (ValueError, TypeError):
        return None

def load_palette(file_path):
    data = pd.read_csv(file_path)
    colors = [hex_to_rgb(color) for color in data['HTML'].dropna() if hex_to_rgb(color) is not None]
    return np.array([color for color in colors if len(color) == 3])

def find_closest_color(color, palette, dark_bias=0):
    distances = np.linalg.norm(palette - np.array(color), axis=1)
    if dark_bias > 0:
        luminance = np.sum(palette, axis=1)
        distances -= dark_bias / (luminance + 1e-6)
    return palette[np.argmin(distances)]

def blend_neighbors(pixels, x, y, size):
    """Blend neighboring pixels to smooth colors."""
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                neighbors.append(tuple(pixels[ny, nx]))
    return tuple(np.mean(neighbors, axis=0).astype(int))

def process_image(image, palette, grid_size, blend_mode="closest", dark_bias=50):
    image = image.resize((grid_size, grid_size), Image.Resampling.LANCZOS)
    pixels = np.array(image)

    output_pixels = np.zeros_like(pixels)
    for y in range(pixels.shape[0]):
        for x in range(pixels.shape[1]):
            if blend_mode == "closest":
                output_pixels[y, x] = find_closest_color(pixels[y, x], palette, dark_bias)
            elif blend_mode == "grayscale":
                grayscale = int(0.299 * pixels[y, x][0] + 0.587 * pixels[y, x][1] + 0.114 * pixels[y, x][2])
                bw_palette = np.array([(0, 0, 0), (255, 255, 255)])
                output_pixels[y, x] = find_closest_color((grayscale, grayscale, grayscale), bw_palette)
            elif blend_mode == "blend":
                blended_color = blend_neighbors(pixels, x, y, grid_size)
                output_pixels[y, x] = find_closest_color(blended_color, palette, dark_bias)
            elif blend_mode == "average":
                neighbors = blend_neighbors(pixels, x, y, grid_size)
                output_pixels[y, x] = find_closest_color(neighbors, palette, dark_bias)
            elif blend_mode == "dominant":
                neighbors = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < grid_size and 0 <= ny < grid_size:
                            neighbors.append(tuple(pixels[ny, nx]))
                dominant_color = Counter(neighbors).most_common(1)[0][0]
                output_pixels[y, x] = find_closest_color(dominant_color, palette, dark_bias)
            elif blend_mode == "weighted":
                weight = 0.6
                neighbors = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < grid_size and 0 <= ny < grid_size:
                            neighbors.append(pixels[ny, nx])
                neighbors = np.array(neighbors)
                blended_color = tuple((weight * pixels[y, x] + (1 - weight) * neighbors.mean(axis=0)).astype(int))
                output_pixels[y, x] = find_closest_color(blended_color, palette, dark_bias)
            elif blend_mode == "contrast_boost":
                factor = 1.5
                boosted_color = tuple(np.clip((pixels[y, x] - 128) * factor + 128, 0, 255).astype(int))
                output_pixels[y, x] = find_closest_color(boosted_color, palette, dark_bias)
    return output_pixels


def render_perler_layout(output_pixels, bead_size, border_size):
    grid_size = output_pixels.shape[0]
    canvas_size = (grid_size * (bead_size + border_size), grid_size * (bead_size + border_size))
    output_image = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(output_image)

    for y in range(grid_size):
        for x in range(grid_size):
            color = tuple(output_pixels[y, x])
            left = x * (bead_size + border_size) + border_size
            top = y * (bead_size + border_size) + border_size
            right = left + bead_size
            bottom = top + bead_size
            draw.rectangle([left, top, right, bottom], fill=color, outline="white")

    return output_image

# Streamlit App
st.title("Perler Bead Layout Generator")

# Upload an image
uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
if uploaded_image is not None:
    input_image = Image.open(uploaded_image).convert("RGB")
    st.image(input_image, caption="Uploaded Image", use_column_width=True)

    # Settings
    grid_size = st.slider("Grid Size", 10, 100, 58)
    bead_size = st.slider("Bead Size", 5, 20, 10)
    border_size = st.slider("Border Size", 0, 5, 1)
 # Blend Mode Options
    blend_mode = st.selectbox("Blend Mode", [
        "closest", 
        "grayscale", 
        "blend", 
        "average", 
        "dominant", 
        "weighted", 
        "contrast_boost"
    ])
    dark_bias = st.slider("Dark Bias", 0, 100, 50)

    # Load the Perler palette
    palette_file = "../../assets/perlercolor - FULL.csv"  # Replace with your CSV file path
    palette = load_palette(palette_file)

    # Process the image
    st.text("Processing image...")
    output_pixels = process_image(input_image, palette, grid_size, blend_mode, dark_bias)

    # Render the layout
    output_image = render_perler_layout(output_pixels, bead_size, border_size)

    # Display the output
    st.image(output_image, caption="Generated Perler Layout", use_column_width=True)

