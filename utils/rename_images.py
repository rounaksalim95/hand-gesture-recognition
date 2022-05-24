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

    # Remove whitespace and other bothersome characters from filenames
    for i in range(len(images)):
        original_image = images[i]
        images[i] = images[i].replace(" ", "")
        images[i] = images[i].replace("(", "_")
        images[i] = images[i].replace(")", "")
        os.rename(f"{directory}/{original_image}", f"{directory}/{images[i]}")

    new_directory = f"{directory}/new"
    # Copy images to new directory with new names
    for i in range(len(images)):
        os.system(f"cp {directory}/{images[i]} {new_directory}/{i + 1}.png")


if __name__ == "__main__":
    app.run(main)
