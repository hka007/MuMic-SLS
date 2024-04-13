import os
import random
import shutil

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def create_test_set():
    # Source and destination directories
    source_directory = '/Volumes/hka/Advision Data/002/Masterarbeit-Hamoud/'
    destination_directory = '/Volumes/hka/Advision Data/test'

    # List all image files in the source directory
    image_files = [f for f in os.listdir(source_directory) if f.endswith(('.jpg', '.png', '.jpeg', '.gif'))]
    print(image_files.count)
    # Check if there are at least 1000 image files
    if len(image_files) < 1000:
        print("Not enough image files in the source directory.")
    else:
        # Randomly select 1000 files
        selected_files = random.sample(image_files, 1000)

        # Create the destination directory if it doesn't exist
        os.makedirs(destination_directory, exist_ok=True)

        # Copy the selected files to the destination directory
        for file_name in selected_files:
            source_file = os.path.join(source_directory, file_name)
            destination_file = os.path.join(destination_directory, file_name)
            shutil.move(source_file, destination_file)
            print(file_name)

        print("Copied 1000 random image files to the destination directory.")

    image_files = [f for f in os.listdir(source_directory) if f.endswith(('.jpg', '.png', '.jpeg', '.gif'))]
    print(image_files.count)
