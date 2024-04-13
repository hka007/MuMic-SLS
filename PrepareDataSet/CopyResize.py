import os
from PIL import Image


def resize_and_copy_images(source_folders, target_folder, size=(244, 244)):
    """
    Copies images from multiple source folders, resizes them to the specified size,
    and saves them to the target folder.

    Parameters:
    - source_folders (list): List of paths to source folders containing images.
    - target_folder (str): Path to the target folder where the transformed images will be saved.
    - size (tuple, optional): Desired size for the transformed images. Default is (244, 244).
    """

    # Create target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for source in source_folders:
        for filename in os.listdir(source):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                img_path = os.path.join(source, filename)
                target_path = os.path.join(target_folder, filename)

                # Check if the image already exists in the target folder
                if os.path.exists(target_path):
                    print(f"Image {filename} already exists in the target folder. Skipping...")
                    continue
                try:
                    with Image.open(img_path) as img:
                        img_resized = img.resize(size)
                        img_resized.save(target_path)
                        print(f"Copied and transformed {filename} to {target_folder}")
                except OSError:
                    print(f"Failed to process {filename}. Possibly a corrupted image.")

#
source_folders = ["/Volumes/hka/Advision Data/001/Masterarbeit-Hamoud/",
    "/Volumes/hka/Advision Data/002/Masterarbeit-Hamoud/"
    , "/Volumes/hka/Advision Data/003/Masterarbeit-Hamoud/"
    , "/Volumes/hka/Advision Data/004/Masterarbeit-Hamoud/"
    , "/Volumes/hka/Advision Data/005/Masterarbeit-Hamoud/"
    , "/Volumes/hka/Advision Data/006/Masterarbeit-Hamoud/"
    , "/Volumes/hka/Advision Data/007/Masterarbeit-Hamoud/"
    ,"/Volumes/hka/Advision Data/008/Masterarbeit-Hamoud/"]
target_folder = "/Users/hka-private/PycharmProjects/Advision/all_transformed_images"
resize_and_copy_images(source_folders, target_folder)
