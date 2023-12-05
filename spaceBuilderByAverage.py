from PIL import Image, ImageDraw
import numpy as np
import os


def colorDetector(color):
    #change this to find the x most common colors and assign them a Random? color from here? 
    colors = [[249,108,98], [245, 125, 32],[251,171,24], [252,195,158],[227,224,41],[0,175,77], [24,158,159], [132,200,226], [0,57,94], [255,255,255]]
    color = np.array(color)
    distances = np.linalg.norm(colors - color, axis=1)
    index_of_smallest = np.argmin(distances)

    return tuple(colors[index_of_smallest])


def make_square(image):
    width, height = image.size

    # Determine the smaller dimension
    min_dim = max(width, height)

    # Calculate cropping coordinates
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = (width + min_dim) // 2
    bottom = (height + min_dim) // 2

    # Crop the image to a square
    square_image = image.crop((left, top, right, bottom))
    return square_image

def cut_into_tiles(img, d):
    width, height = img.size
    tile_width = width // d
    tile_height = height // d
    tiles = []

    for i in range(d):
        for j in range(d):
            left = i * tile_width
            upper = j * tile_height
            right = (i + 1) * tile_width
            lower = (j + 1) * tile_height
            tile = img.crop((left, upper, right, lower))
            tiles.append(tile)

    return tiles


def convert_to_solid_color(tiles):
    solid_color_tiles = []

    for tile in tiles:
        # Convert 'P' mode images to grayscale
        if tile.mode == 'P':
            tile = tile.convert('L')

        np_tile = np.array(tile)

        # Now tile is either 'L' (grayscale) or 'RGB'
        if tile.mode == 'L':  # Grayscale image
            average_intensity = int(np_tile.mean())
            # Convert the intensity to an RGB tuple
            average_color = (average_intensity, average_intensity, average_intensity)
        elif tile.mode == 'RGB':  # RGB image
            average_color = tuple(np_tile.mean(axis=(0, 1)).astype(int))
        else:
            raise ValueError(f"Unsupported image mode: {tile.mode}")

        legoColor = colorDetector(average_color)

#go for a random change if there is a "large" color difference between

        solid_color_tile = Image.new('RGB', tile.size, legoColor)
        solid_color_tiles.append(solid_color_tile)

    return solid_color_tiles


def combine_tiles(tiles, d):
    if not tiles:
        raise ValueError("Tile list is empty.")

    tile_width, tile_height = tiles[0].size
    new_image = Image.new('RGB', (tile_width * d, tile_height * d))

    for i in range(d):
        for j in range(d):
            tile = tiles[i * d + j]
            new_image.paste(tile, (i * tile_width, j * tile_height))

    return new_image

def crop_to_circle(tiles, inset_factor):
    circular_tiles = []
    for tile in tiles:
        mask = Image.new('L', tile.size, 0)
        mask_draw = ImageDraw.Draw(mask)

        width, height = tile.size
        inset_width = width * inset_factor
        inset_height = height * inset_factor

        ellipse_bounds = (
            inset_width, inset_height, 
            width - inset_width, height - inset_height
        )

        mask_draw.ellipse(ellipse_bounds, fill=255)

        result = Image.new('RGB', tile.size, (0, 0, 0))
        result.paste(tile, mask=mask)
        circular_tiles.append(result)

    return circular_tiles

def createLegoTiles(path):
    try:
        d_large = 5 
        d_small = 16

        image = Image.open(filepath)

        image = make_square(image)

        large_tiles = cut_into_tiles(image, d_large)

        final_tiles = []

        # Process each large tile
        for i, large_tile in enumerate(large_tiles):
            small_tiles = cut_into_tiles(large_tile, d_small)
            solid_color_tiles = convert_to_solid_color(small_tiles)
            circle_solid_color = crop_to_circle(solid_color_tiles, 0.10)
            combined_small_tile = combine_tiles(circle_solid_color, d_small)

            row = i // d_large  # Integer division for row number
            col = i % d_large   # Remainder for column number
            
            label = f"{row + 1}x{col + 1}"  # Adding 1 to start labeling from 1 instead of 0
            combined_small_tile.save(os.path.join('./output_folder', f"tile_{label}.png"))

            final_tiles.append(combined_small_tile)

        # Combine the final tiles into the original image layout
        combined_image = combine_tiles(final_tiles, d_large)
        combined_image.show()

    except IOError as e:
        print(f"An error occurred while opening the image: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

filepath = "./masterSword.png"

createLegoTiles(filepath)