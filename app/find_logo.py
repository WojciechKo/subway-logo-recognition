import sys
import os
import shutil

from image_analizer import ImageAnalizer
from invariants_calculator import InvariantsCalculator

images_dir = sys.argv[1]
IMAGES_DIRECTORY = os.path.join(os.getcwd(), images_dir)

def create_subdirectory_for_each_image():
    images = [os.path.join(IMAGES_DIRECTORY, file) for file in os.listdir(IMAGES_DIRECTORY) if not file.startswith('.')]
    images = [file for file in images if os.path.isfile(file)]

    for image_path in images:

        image_dir, image_extension = os.path.splitext(image_path)

        if os.path.exists(image_dir):
            shutil.rmtree(image_dir, ignore_errors=True)
        os.makedirs(image_dir)
        shutil.copyfile(image_path, os.path.join(image_dir, "0" + image_extension))

def analize_images():
    image_dirs = [os.path.join(IMAGES_DIRECTORY, file) for file in os.listdir(IMAGES_DIRECTORY)]
    image_dirs = [file for file in image_dirs if os.path.isdir(file)]

    for image_dir in image_dirs:
        image = os.listdir(image_dir)[0]
        image_path = os.path.join(image_dir, image)
        ImageAnalizer.analize(image_path)

create_subdirectory_for_each_image()

InvariantsCalculator().calculate(os.path.join(IMAGES_DIRECTORY, "subway_logo/0.png"))

analize_images()
