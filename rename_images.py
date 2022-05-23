"""
Script to rename the images in a given directory
"""

import os

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("directory", "", "Directory to rename images")

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

    # Rename images
    for i in range(len(images)):
        os.rename(f"{directory}/{images[i]}", f"{directory}/{i + 1}.png")

if __name__ == '__main__':
    app.run(main)
