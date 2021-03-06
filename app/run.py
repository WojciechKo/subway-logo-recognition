import os
import sys
import shutil

from image_analizer import ImageAnalizer
from subway_invariants_model import SubwayInvariantsModel

import logging

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('execution.log')
fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

logger.addHandler(fh)
logger.addHandler(ch)

images_dir = sys.argv[1] if len(sys.argv) > 1 else 'images'

IMAGES_DIR = os.path.join(os.getcwd(), images_dir)
TEST_IMAGES_DIR = os.path.join(IMAGES_DIR, "test")

def create_subdirectory_for_each_test_image():
    images = [os.path.join(TEST_IMAGES_DIR, file) for file in os.listdir(TEST_IMAGES_DIR) if not file.startswith('.')]
    images = [file for file in images if file.endswith(".png")]

    for image_path in images:
        image_dir, _image_extension = os.path.splitext(image_path)

        if os.path.exists(image_dir):
            shutil.rmtree(image_dir, ignore_errors=True)
        os.makedirs(image_dir)
        shutil.copyfile(image_path, os.path.join(image_dir, "0_original.png"))

def analize_images(model):
    image_dirs = [os.path.join(TEST_IMAGES_DIR, file) for file in os.listdir(TEST_IMAGES_DIR)]
    image_dirs = [file for file in image_dirs if os.path.isdir(file)]

    for image_dir in image_dirs:
        image_path = os.path.join(image_dir, '0_original.png')
        ImageAnalizer(model).analize(image_path)

create_subdirectory_for_each_test_image()
model = SubwayInvariantsModel(os.path.join(IMAGES_DIR, "logo", "subway.png"))
analize_images(model)
