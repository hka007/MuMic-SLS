import os
import pandas as pd


def get_image_files(directory_path):
    # Define a set of image extensions to search for
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

    # Use a list comprehension to filter the files in the directory
    return [file for file in os.listdir(directory_path) if os.path.splitext(file)[1].lower() in image_extensions]


def build_multilabel(df, column, labels):
    df_dummies = pd.get_dummies(df[column])
    # Add the 'id_centralads' column to the dummies dataframe
    df_dummies['id_centralads'] = df['id_centralads']
    # Group by 'id_centralads' and sum the dummy columns
    df_grouped = df_dummies.groupby('id_centralads').sum()
    # Ensure that the values are either 0 or 1 (in case of multiple rows with the same rubricname for an id)
    df_grouped = (df_grouped > 0).astype(int)
    # Ensure the correct column order based on all_labels
    for label in labels:
        if label not in df_grouped.columns:
            df_grouped[label] = 0
    df_grouped = df_grouped[labels]
    # Reset the index
    df_grouped.reset_index(inplace=True)
    # Create the 'labels_rubricname' column
    df_grouped['labels'] = df_grouped[labels].values.tolist()
    # Group by 'id_centralads' in the original dataframe and aggregate captions
    captions_agg = df.groupby('id_centralads')[column].apply(lambda x: ','.join(set(x))).reset_index().astype(str)

    # Add the 'caption' column by merging with the original dataframe on 'id_centralads'
    df_grouped = df_grouped.merge(captions_agg, on='id_centralads', how='left')
    df_grouped.rename(columns={column: 'caption'}, inplace=True)

    # Drop the one-hot encoded columns
    df_grouped = df_grouped[['id_centralads', 'labels', 'caption']]

    df_grouped['id_centralads'] = df_grouped['id_centralads'].astype(str)
    df_grouped['id_centralads'] = df_grouped['id_centralads'] + '.jpg'
    df_grouped.rename(columns={'id_centralads': 'image'}, inplace=True)

    # Step 2: Reset the index
    df_grouped.reset_index(inplace=True)

    # Step 3: Rename the new column to 'id'
    df_grouped.rename(columns={'index': 'id'}, inplace=True)

    print(df_grouped.head(2))

    return df_grouped


prefix = ""
#directory_path = "/Volumes/hka/Advision Data/test"
directory_path = "/Users/hka-private/PycharmProjects/Advision/all_transformed_images"
images = get_image_files(directory_path)
motive_zu_design_df = pd.read_csv('/Volumes/hka/Advision Data/2022-10-10_Motive_zu_Designcodes.csv', delimiter=';',
                                  quotechar='"').astype(str)

image_df = pd.DataFrame(images, columns=['id_centralads']).astype(str)
image_df['id_centralads'] = image_df['id_centralads'].str.replace('.jpg', '', regex=False)
image_df['id_centralads'] = image_df['id_centralads'].astype(str)
image_df.to_csv('../AdvisionDataset250/image_names.csv', index=False)

motive_zu_design_df['id_centralads'] = motive_zu_design_df['id_centralads'].astype(float).astype('Int64').astype(str)
motive_zu_design_df = motive_zu_design_df.drop(columns=['id_rubric', 'id_aspect', 'id_element'])
filtered_motive_zu_design_df = motive_zu_design_df[motive_zu_design_df['id_centralads'].isin(image_df['id_centralads'])]
filtered_motive_zu_design_df = filtered_motive_zu_design_df.drop(columns=['id'])

#######################################################################################################################
'''
multilabel_rubric
all_rubric_labels

multilabel_rubric_aspect
all_rubric_aspect_labels

multilabel_rubric_aspect_element
all_rubric_aspect_element_labels
'''
#######################################################################################################################
'''create multilabel for rubric'''
rubric_df = filtered_motive_zu_design_df
rubric_df['rubric'] = rubric_df.apply(
    lambda row: f"{row['rubricname']}", axis=1)
rubric_aspect_labels = list(set(rubric_df['rubric']))
df = pd.DataFrame(rubric_df)
filtered_motive_zu_design_rubric_aspect_df = list(set(rubric_df['rubric']))
all_labels = pd.DataFrame(filtered_motive_zu_design_rubric_aspect_df)
all_labels.to_csv(f'../AdvisionDataset250/{prefix}_all_labels_rubric.csv', index=False)
multilabel_rubricname_df = build_multilabel(df, 'rubric', rubric_aspect_labels)
multilabel_rubricname_df.to_csv(f'../AdvisionDataset250/{prefix}_multilabel_rubric.csv', index=False)

#######################################################################################################################
'''create multilabel for rubric_aspect'''
rubric_aspect_df = filtered_motive_zu_design_df
rubric_aspect_df['rubric_aspect'] = rubric_aspect_df.apply(
    lambda row: f"{row['rubricname']} {row['aspectname']}", axis=1)
rubric_aspect_labels = list(set(rubric_aspect_df['rubric_aspect']))
df = pd.DataFrame(rubric_aspect_df)
filtered_motive_zu_design_rubric_aspect_df = list(set(rubric_aspect_df['rubric_aspect']))
all_labels = pd.DataFrame(filtered_motive_zu_design_rubric_aspect_df)
all_labels.to_csv(f'../AdvisionDataset250/{prefix}_all_labels_rubric_aspect.csv', index=False)
multilabel_rubricname_df = build_multilabel(df, 'rubric_aspect', rubric_aspect_labels)
multilabel_rubricname_df.to_csv(f'../AdvisionDataset250/{prefix}_multilabel_rubric_aspect.csv', index=False)

#######################################################################################################################
'''create multilabel for rubric_aspect_element'''
rubric_aspect_element_df = filtered_motive_zu_design_df
rubric_aspect_element_df['rubric_aspect_element'] = rubric_aspect_element_df.apply(
    lambda row: f"{row['rubricname']} {row['aspectname']} {row['elementname']}", axis=1)
rubric_aspect_labels = list(set(rubric_aspect_element_df['rubric_aspect_element']))
df = pd.DataFrame(rubric_aspect_element_df)
filtered_motive_zu_design_rubric_aspect_df = list(set(rubric_aspect_element_df['rubric_aspect_element']))
all_labels = pd.DataFrame(filtered_motive_zu_design_rubric_aspect_df)
all_labels.to_csv(f'../AdvisionDataset250/{prefix}_all_labels_rubric_aspect_element.csv', index=False)
multilabel_rubricname_df = build_multilabel(df, 'rubric_aspect_element', rubric_aspect_labels)
multilabel_rubricname_df.to_csv(f'../AdvisionDataset250/{prefix}_multilabel_rubric_aspect_element.csv', index=False)

#######################################################################################################################
'''create '''
rubric_aspect_element_df = filtered_motive_zu_design_df
rubric_aspect_element_df['rubric_aspect_element'] = rubric_aspect_element_df.apply(
    lambda row: f"{row['rubricname']} {row['aspectname']} {row['elementname']}", axis=1)
rubric_aspect_labels = list(set(rubric_aspect_element_df['rubric_aspect_element']))
df = pd.DataFrame(rubric_aspect_element_df)
filtered_motive_zu_design_rubric_aspect_df = list(set(rubric_aspect_element_df['rubric_aspect_element']))
all_labels = pd.DataFrame(filtered_motive_zu_design_rubric_aspect_df)
all_labels.to_csv(f'../AdvisionDataset250/{prefix}_all_labels_rubric_aspect_element.csv', index=False)
multilabel_rubricname_df = build_multilabel(df, 'rubric_aspect_element', rubric_aspect_labels)
multilabel_rubricname_df.to_csv(f'../AdvisionDataset250/{prefix}_multilabel_rubric_aspect_element.csv', index=False)

