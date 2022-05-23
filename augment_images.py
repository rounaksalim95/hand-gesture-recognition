"""
Script to augment images in a given directory by flipping them horizontally
"""

import os

from absl import app, flags
from PIL import Image, ImageOps

FLAGS = flags.FLAGS

flags.DEFINE_string("directory", "", "Directory to augment images")

def main(argv):
    directory = ""

    if FLAGS.directory:
        directory = FLAGS.directory

    if not directory:
        raise ValueError("Directory not specified")

    if not os.path.exists(directory):
        raise ValueError("Directory does not exist")

    images = []

    # Get list of images in directory
    for file in os.listdir(directory):
        if file.endswith(".png"):
            images.append(file)

    # Augment images
    for i in range(len(images)):
        image = Image.open(f"{directory}/{images[i]}")
        image = ImageOps.mirror(image)
        image.save(f"{directory}/{i + 1}-flipped.png")


if __name__ == '__main__':
    app.run(main)
