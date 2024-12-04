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
    """Load Perler bead palette, clean hex values, filter by BRAND, and return as DataFrame and RGB array."""
    data = pd.read_csv(file_path)

    # Filter rows where the BRAND is 'PERLER'
    data = data[data['BRAND'].str.upper() == 'PERLER']

    # Clean hex values: Ensure they start with '#' and are uppercase
    def clean_hex(hex_code):
        if isinstance(hex_code, str):
            if not hex_code.startswith('#'):
                hex_code = f'#{hex_code}'  # Add '#' if missing
            return hex_code.upper()  # Convert to uppercase
        return None

    data['HTML'] = data['HTML'].apply(clean_hex)
    data = data.dropna(subset=['HTML'])  # Drop invalid or missing hex values

    # Convert valid hex codes to RGB tuples
    data['RGB'] = data['HTML'].apply(lambda hex_code: hex_to_rgb(hex_code))

    # Return the cleaned DataFrame and the RGB array
    return data, np.array(data['RGB'].tolist())


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
            # Add other blend modes as needed here
    return output_pixels


def render_perler_layout(output_pixels, bead_size, border_size):
    """Render a plain Perler bead layout without numbers."""
    grid_size = output_pixels.shape[0]
    canvas_width = grid_size * (bead_size + border_size)
    canvas_height = grid_size * (bead_size + border_size)

    # Create a blank canvas
    output_image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(output_image)

    for y in range(grid_size):
        for x in range(grid_size):
            color = tuple(output_pixels[y, x])
            left = x * (bead_size + border_size) + border_size
            top = y * (bead_size + border_size) + border_size
            right = left + bead_size
            bottom = top + bead_size

            # Draw each bead as a rectangle
            draw.rectangle([left, top, right, bottom], fill=color, outline="white")

    return output_image


def render_perler_layout_with_numbers(output_pixels, bead_size, border_size, palette_data):
    """Render the Perler bead layout with numbered labels."""
    grid_size = output_pixels.shape[0]
    canvas_width = grid_size * (bead_size + border_size)
    canvas_height = grid_size * (bead_size + border_size)

    output_image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(output_image)

    legend = {}
    current_number = 1

    for y in range(grid_size):
        for x in range(grid_size):
            color = tuple(output_pixels[y, x])
            hex_color = '#{:02X}{:02X}{:02X}'.format(*color)  # Convert RGB to HEX format

            matched_row = palette_data[palette_data['HTML'] == hex_color]
            if not matched_row.empty:
                color_name = matched_row.iloc[0]['NAME']
            else:
                color_name = "Unknown"

            if hex_color not in legend:
                legend[hex_color] = (current_number, color_name)
                current_number += 1

            number_label = legend[hex_color][0]

            left = x * (bead_size + border_size) + border_size
            top = y * (bead_size + border_size) + border_size
            right = left + bead_size
            bottom = top + bead_size

            draw.rectangle([left, top, right, bottom], fill=color, outline="white")
            text_x = (left + right) // 2
            text_y = (top + bottom) // 2
            draw.text((text_x, text_y), str(number_label), fill="black", anchor="mm")

    return output_image, legend


def generate_legend_image(legend, color_counts, bead_size=20):
    """Generate a legend image mapping numbers to color names with usage count."""
    rows = len(legend)
    canvas_height = rows * bead_size
    canvas_width = 500  # Adjust width to fit text comfortably

    legend_image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(legend_image)

    y_offset = 0
    for hex_color, (number, color_name) in legend.items():
        color = tuple(int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        count = color_counts.get(hex_color, 0)  # Get the usage count of the color

        left = 5
        top = y_offset
        right = left + bead_size
        bottom = top + bead_size

        # Draw the color swatch
        draw.rectangle([left, top, right, bottom], fill=color, outline="black")

        # Draw the text: Number, Color Name, Hex, and Count
        text = f"{number}: {color_name} ({hex_color}) - {count} uses"
        draw.text((right + 10, top + bead_size // 2), text, fill="black", anchor="lm")

        y_offset += bead_size

    return legend_image


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
    blend_mode = st.selectbox("Blend Mode", ["closest", "grayscale", "blend"])
    show_numbers = st.checkbox("Show Numbers on Layout", value=True)

    # Load the Perler palette
    palette_file = "../../assets/perlercolor - FULL.csv"  # Replace with your CSV file path
    palette_data, target_colors = load_palette(palette_file)

    # Process the image
    output_pixels = process_image(input_image, target_colors, grid_size, blend_mode)

# Generate color counts
    unique_colors, counts = np.unique(output_pixels.reshape(-1, output_pixels.shape[2]), axis=0, return_counts=True)
    color_counts = {f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}": count for color, count in zip(unique_colors, counts)}

    if show_numbers:
        output_image, legend = render_perler_layout_with_numbers(output_pixels, bead_size, border_size, palette_data)
        st.image(output_image, caption="Generated Perler Layout with Numbers", use_container_width=True)

        # Generate and display the legend with usage counts
        legend_image = generate_legend_image(legend, color_counts)
        st.image(legend_image, caption="Legend with Counts", use_column_width=False)
    else:
        output_image = render_perler_layout(output_pixels, bead_size, border_size)
        st.image(output_image, caption="Generated Perler Layout without Numbers", use_column_width=True)
