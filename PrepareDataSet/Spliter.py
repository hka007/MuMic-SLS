import random

import pandas as pd
import os
def get_image_files(directory_path):
    # Define a set of image extensions to search for
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

    # Use a list comprehension to filter the files in the directory
    return [file for file in os.listdir(directory_path) if os.path.splitext(file)[1].lower() in image_extensions]

labels_df = pd.read_csv('/Volumes/hka/Advision Data/2022-10-10_Motive_zu_Designcodes.csv',delimiter=';', quotechar='"').astype(str)
labels_df['id_centralads'] = labels_df['id_centralads'].astype(float).astype('Int64').astype(str)
filtered_df = labels_df
filtered_df['caption_number'] = labels_df.groupby('id_centralads').cumcount()
filtered_df['caption'] = labels_df['rubricname'] + ' ist ' + labels_df['aspectname'] + ' und ist ' + labels_df['elementname']

captions = set(filtered_df['caption'])
selected_items = random.sample(captions, 80)


directory_path = "/Users/hka-private/PycharmProjects/Advision/all_transformed_images"
images = get_image_files(directory_path)

image_df = pd.DataFrame(images, columns=['id_centralads']).astype(str)
image_df['id_centralads'] = image_df['id_centralads'].str.replace('.jpg', '', regex=False)
image_df['id_centralads'] = image_df['id_centralads'].astype(str)
image_df.to_csv('../AdvisionDataset250/image_names.csv', index=False)

print(filtered_df.head())