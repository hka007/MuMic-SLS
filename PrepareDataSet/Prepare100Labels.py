import os
import pandas as pd
from sklearn.model_selection import train_test_split


def get_image_files(directory_path):
    # Define a set of image extensions to search for
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

    # Use a list comprehension to filter the files in the directory
    return [file for file in os.listdir(directory_path) if os.path.splitext(file)[1].lower() in image_extensions]


def combine_labels(row):
    return f"{row['rubricname']} ist {row['aspectname']} und ist {row['elementname']}".replace(",", "").strip()


def first():
    labels_df = pd.read_csv('/Volumes/hka/Advision Data/2022-10-10_Motive_zu_Designcodes.csv', delimiter=';',
                            quotechar='"').astype(str)
    labels_df['id_centralads'] = labels_df['id_centralads'].astype(float).astype('Int64').astype(str)
    filtered_df = labels_df.drop(columns=['id'])

    filtered_df['caption_number'] = filtered_df.groupby('id_centralads').cumcount()
    filtered_df['caption'] = filtered_df['rubricname'] + ' ist ' + filtered_df['aspectname'] + ' und ist ' + \
                             filtered_df[
                                 'elementname']

    captions_df = filtered_df['caption'].drop_duplicates().reset_index(drop=True)

    # captions_df = captions_df.sample(n=100)

    captions_df.to_csv("../AdvisionDatasetAllData/captions.csv", index=False)

    print(filtered_df.head())


################################################################################################################################################
################################################ End Create 100 Labels #########################################################################
################################################################################################################################################

def second():
    ## read images from motive_zu_designcode
    Motive_zu_Designcodes = pd.read_csv('/Volumes/hka/Advision Data/2022-10-10_Motive_zu_Designcodes.csv',
                                        delimiter=';',
                                        quotechar='"').astype(str)

    Motive_zu_Designcodes['combined'] = Motive_zu_Designcodes.apply(combine_labels, axis=1)
    #### now Motive_zu_Designcodes has combined ####
    labels_list = pd.read_csv("../AdvisionDatasetAllData/captions.csv")
    labels_list = list(labels_list['caption'])
    filtered_Motive_zu_Designcodes = Motive_zu_Designcodes[Motive_zu_Designcodes['combined'].isin(labels_list)]

    df = filtered_Motive_zu_Designcodes.groupby("id_centralads")['combined'].apply(lambda x: ','.join(x)).reset_index()
    df = df.rename(columns={'id_centralads': 'image'})
    df['image'] = df['image'].apply(lambda x: x.replace(".0", ".jpg"))

    df = df.rename(columns={'combined': 'caption'})

    df = df[:-1]

    # df['id'] = df['image'].factorize()[0]

    df.to_csv('../AdvisionDatasetAllData/data_advision.csv', index=False)

    # Resulting DataFrame with desired images
    print(df)


################################################################################################################################################
################################################ End Create data_advision #########################################################################
# 9996966.jpg,Menschen ist Gruppe/Paar und ist Mann/Frau #
# 9997246.jpg,Menschen ist Ohne Kontext und ist Mann #
################################################################################################################################################

def third():
    directory_path = "/Users/hka-private/PycharmProjects/Advision/all_transformed_images"
    images = get_image_files(directory_path)

    labels = list(pd.read_csv("../AdvisionDatasetAllData/captions.csv")['caption'])
    data_advision = pd.read_csv("../AdvisionDatasetAllData/data_advision.csv")

    label_index = {label: idx for idx, label in enumerate(labels)}

    filtered_df = data_advision[data_advision['image'].isin(images)]

    multilabel_dict = {image: [0] * len(labels) for image in filtered_df['image'].unique()}

    rows = []

    # Populate the multilabel vectors
    for _, row in filtered_df.iterrows():
        image = row['image']
        labels = row['caption']
        for label in labels.split(','):
            label = label.strip()
            multilabel_dict[image][label_index[label]] = 1

        row = {'image': image, 'labels': multilabel_dict[image], 'caption': labels}
        rows.append(row)

    df = pd.DataFrame(rows)
    df['id'] = df.index

    df.to_csv("../AdvisionDatasetAllData/Advision_ML.csv", index=False)


################################################################################################################################################
################################################ Spliter #########################################################################

def forth():
    df = pd.read_csv("../AdvisionDatasetAllData/Advision_ML.csv")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Saving the datasets into CSV files
    train_df.to_csv('../AdvisionDatasetAllData/train_dataset.csv', index=False)
    test_df.to_csv('../AdvisionDatasetAllData/test_dataset.csv', index=False)


# first()
# second()
# third()
forth()