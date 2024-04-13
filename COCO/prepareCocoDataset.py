import pandas as pd
from pycocotools.coco import COCO

annotations_path = '/Users/hka-private/PycharmProjects/Dataset/annotations/instances_val2017.json'

# Load COCO annotations (modify paths accordingly)
coco = COCO(annotations_path)  # Path to COCO annotations file

# Create a CSV file to store the data
csv_file = 'test_coco_multilabel_classification.csv'

# Initialize an empty list to store data
data = []

all_labels = [cat['name'] for cat in coco.loadCats(coco.getCatIds())]
df = pd.DataFrame(all_labels)
df.to_csv("test_coco_all_labels.csv", index=False)

# Create a mapping from category names to category IDs
category_name_to_id = {cat['name']: i for i, cat in enumerate(coco.loadCats(coco.getCatIds()))}

# Determine the maximum category ID in the dataset
max_category_id = max(category_name_to_id.values())

# Initialize the labels list with the correct length
labels = [0] * len(category_name_to_id.values())

# Iterate through COCO annotations
for img_id in coco.getImgIds():
    img = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(ann_ids)

    # Reset labels to all zeros for each image
    labels = [0] * len(category_name_to_id.values())

    # Set labels to 1 for categories present in the image
    caption = []
    for ann in anns:
        print(coco.loadCats(ann['category_id'])[0])
        category_name = coco.loadCats(ann['category_id'])[0]['name']
        category_id = category_name_to_id.get(category_name, None)
        if category_id is not None:
            labels[category_id] = 1  # Subtract 1 because category IDs are 1-based
            caption.append(category_name)

    # Create a row for the CSV
    row = {
        'image': img['file_name'],
        'labels': labels,
        'caption': ','.join(set(caption))
    }
    data.append(row)

# Add an index column named "id"

# Create a DataFrame and save to CSV

df = pd.DataFrame(data)
df = df[df['caption'].notna() & (df['caption'] != '')]

df.reset_index(inplace=True)
df.rename(columns={'index': 'id'}, inplace=True)

#df.to_csv(csv_file, index=False)
