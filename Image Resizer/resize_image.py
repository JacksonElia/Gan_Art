import os
import numpy as np
from PIL import Image


def resize_images(input_path: str, output_binary_path: str, resize=128, channels=3):
    """
    This function will convert a directory of images to the specified size (a square)
    :param input_path: the path to the directory of images
    :param output_binary_path: the path to where the np binary is stored
    :param resize: the side length of the square you want the images to be converted to
    :param channels: amount of channels in NP array
    """

    # These are settings for the image resizing
    training_images = []

    # Goes through all of the images and resizes them to 128x128px
    for filename in os.listdir(input_path):
        training_images.append(Image.open(os.path.join(input_path, filename)).resize((resize, resize), Image.ANTIALIAS))

    training_images = np.reshape(training_images, (-1, resize, resize, channels))
    training_images = training_images / 127.5 - 1

    np.save(output_binary_path, training_images)
