import os
import pandas as pd


## read the image name from the directory '''/Volumes/hka/Advision Data/001/Masterarbeit-Hamoud'''
## save them in a list of df
## for each image_name get the corresponding labels from motive_zu_designcodes
## create df [image][caption_number][caption][id]


## read the image name from the directory '''/Volumes/hka/Advision Data/001/Masterarbeit-Hamoud'''
def get_image_files(directory_path):
    # Define a set of image extensions to search for
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

    # Use a list comprehension to filter the files in the directory
    return [file for file in os.listdir(directory_path) if os.path.splitext(file)[1].lower() in image_extensions]

directory_path = "/Users/hka-private/PycharmProjects/Advision/all_transformed_images"
images = get_image_files(directory_path)

# Create a DataFrame from the list of image names
image_df = pd.DataFrame(images, columns=['id_centralads']).astype(str)
image_df['id_centralads'] = image_df['id_centralads'].str.replace('.jpg', '', regex=False)
image_df['id_centralads'] = image_df['id_centralads'].astype(str)
image_df.to_csv('../AdvisionDataset250/image_names.csv', index=False)

labels_df = pd.read_csv('/Volumes/hka/Advision Data/2022-10-10_Motive_zu_Designcodes.csv',delimiter=';', quotechar='"').astype(str)
labels_df['id_centralads'] = labels_df['id_centralads'].astype(float).astype('Int64').astype(str)
filtered_df = labels_df[labels_df['id_centralads'].isin(image_df['id_centralads'])]
filtered_df = filtered_df.drop(columns=['id'])
filtered_df['caption_number'] = filtered_df.groupby('id_centralads').cumcount()
filtered_df['caption'] = filtered_df['rubricname'] + ' ist ' + filtered_df['aspectname'] + ' und ist ' + filtered_df['elementname']
filtered_df = filtered_df.drop(columns=['id_rubric', 'id_aspect', 'id_element'])
filtered_df = filtered_df.drop(columns=['rubricname', 'aspectname', 'elementname'])
filtered_df = filtered_df.rename(columns={'id_centralads': 'image'})
filtered_df['image'] = filtered_df['image'] + ".jpg"
filtered_df['id'] = filtered_df['image'].factorize()[0]

filtered_df.to_csv('../AdvisionDataset250/data_advision.csv', index=False)

captions_df = filtered_df['caption'].drop_duplicates().reset_index(drop=True)
captions_df.to_csv("../AdvisionDataset250/captions.csv", index=False)




print(filtered_df.head())



