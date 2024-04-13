import numpy as np
import pandas as pd
import torch
from CLIPDataset import get_transforms, CLIPDataset, CLIPDataset_2,CLIPDataset_3
from config import CFG


def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{CFG.captions_path}")
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    image_ids = np.arange(0, max_id)

    # Calculate the size for validation set
    valid_size = int(0.2 * len(image_ids))

    # Slice image_ids for validation and training
    valid_ids = image_ids[:valid_size]
    train_ids = image_ids[valid_size:]

    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)

    return train_dataframe, valid_dataframe


def make_train_dfs(caption_path):
    dataframe = pd.read_csv(f"{caption_path}")
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    image_ids = np.arange(0, max_id)

    train_dataframe = dataframe[dataframe["id"].isin(image_ids)].reset_index(drop=True)

    return train_dataframe


def build_loaders(dataframe, mode, is_eval=False):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset_2(
        dataframe["image"].values,
        dataframe["labels"],
        transforms=transforms,
        is_eval=is_eval
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader
