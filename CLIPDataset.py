import ast

import albumentations as A
import torch
import cv2
from config import CFG
import random


def parse_string_to_tensor(s):
    # Remove brackets and split the string by comma
    s = s.strip('[]').split(',')
    # Convert the string elements to float and create a tensor
    tensor = torch.tensor([float(x) for x in s], dtype=torch.float32)
    return tensor


def selective_language_supervision(batch_size, alpha=3):
    num_classes = batch_size.size()[0]
    selected_labels = torch.zeros_like(batch_size)
    for i in range(batch_size):
        # Identifying positive samples for the current instance
        Spos = (batch_size[i] == 1).nonzero(as_tuple=True)[0].tolist()

        # Identifying negative samples for the current instance
        Sneg = [idx for idx in range(num_classes) if idx not in Spos]

        # Calculating the number of negative samples to be selected
        num_neg_samples = min(alpha * len(Spos), len(Sneg))

        # Capping the number of negative samples to the available negatives
        num_neg_samples = min(num_neg_samples, len(Sneg))

        # Randomly selecting negative samples
        Sslt = random.sample(Sneg, int(num_neg_samples)) if num_neg_samples > 0 else []

        # Combining positive and selected negative samples
        S = Spos + Sslt

        # Marking selected labels for training
        selected_labels[i, S] = 1
    return selected_labels


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, targets, encoded_labels, tokenizer, transforms, is_eval=False):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.targets = list(targets)
        self.encoded_captions = encoded_labels
        # self.encoded_captions = tokenizer(
        #     list(captions), padding=True, truncation=True, max_length=CFG.max_length
        # )
        self.transforms = transforms

        if is_eval:
            self.image_path = CFG.eval_path
        else:
            self.image_path = CFG.train_path

    def __getitem__(self, idx):

        item = {
            key: values.clone().detach()
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{self.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]
        item['labels'] = self.targets[idx]

        return item

    def __len__(self):
        return len(self.captions)

    def get_one_item(self):
        item = {
            key: values.clone().detach().unsqueeze(0)
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(self.image_filenames)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)
        item['caption'] = self.captions
        item['labels'] = self.targets

        return item


class CLIPDataset_2(torch.utils.data.Dataset):
    def __init__(self, image_filenames, targets, transforms, is_eval=False):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """
        self.image_filenames = image_filenames
        # self.captions = list(captions)
        self.targets = list(targets)

        self.transforms = transforms

        if is_eval:
            self.image_path = CFG.eval_path
        else:
            self.image_path = CFG.train_path

    def __getitem__(self, idx):

        item = {
            # key: torch.tensor(values[idx])
            # for key, values in self.encoded_captions.items()
            # key: values.clone().detach()
            # for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{self.image_path}/{self.image_filenames[idx]}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['labels'] = self.targets[idx]

        return item

    def __len__(self):
        return len(self.targets)

    def get_one_item(self):
        item = {
            # key: torch.tensor(values).to(CFG.device)
            # for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(self.image_filenames)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)
        # item['caption'] = self.captions
        item['labels'] = self.targets

        return item


class CLIPDataset_3(torch.utils.data.Dataset):
    def __init__(self, image_filenames, targets, transforms, is_eval=False):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """
        self.image_filenames = image_filenames
        # self.captions = list(captions)
        self.targets  = list(targets)
        # self.targets = torch.tensor([parse_string_to_tensor_2(s) for s in targets])

        self.transforms = transforms

        self.image_path = CFG.train_path

    def __getitem__(self, idx):
        item = {
            # key: torch.tensor(values[idx])
            # for key, values in self.encoded_captions.items()
            # key: values.clone().detach()
            # for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{self.image_path}/{self.image_filenames[idx]}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['labels'] = self.targets[idx]

        return item

    def __len__(self):
        return len(self.targets)

    def get_one_item(self):
        item = {
            key: torch.tensor(values).to(CFG.device)
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(self.image_filenames)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)
        item['caption'] = self.captions
        item['labels'] = self.targets

        return item


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
