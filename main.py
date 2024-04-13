import os
import torch
import warnings

from matplotlib import pyplot as plt

import Evaluation
import pandas as pd

from config import CFG
from Utils import AvgMeter, get_lr
from tqdm.autonotebook import tqdm
from ResultReport import ResultReport
from transformers import DistilBertTokenizer
from Train import build_loaders, make_train_dfs, make_train_valid_dfs

from MuMicSLS import CLIPModel

warnings.filterwarnings("ignore")

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

if CFG.debug:
    CFG.device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
else:
    CFG.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def get_encoded_label(label_path):
    all_labels_df = pd.read_csv(label_path)
    all_labels = all_labels_df.values

    all_labels_train = [item for sublist in all_labels for item in sublist]
    all_labels_train = [cap.lower() for cap in all_labels_train]

    # all_labels_train = [f"this image contains {word.replace(',', ' and ')}" for word in all_labels_train]

    number_of_labels = len(all_labels_train)
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_labels = tokenizer(all_labels_train, padding=True, truncation=True, return_tensors='pt',
                               max_length=CFG.max_length)

    return encoded_labels


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(train_loader, total=len(train_loader), colour="red")
    encoded_label = get_encoded_label(CFG.all_label)
    for batch in tqdm_object:
        batch = {k: (v.to(CFG.device) if k not in ["caption", "labels"] else v) for k, v in batch.items()}

        loss, _, _ = model(batch, encoded_label)  #

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter.avg


def valid_epoch(model, valid_loader):
    predicted_patch = []
    ground_truth_labels_patch = []
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader), colour="green")
    encoded_label = get_encoded_label(CFG.all_label)

    for batch in tqdm_object:
        # batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        batch = {k: (v.to(CFG.device) if k not in ["caption", "labels"] else v) for k, v in batch.items()}

        loss, logits, targets = model(batch, encoded_label)  #
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

        predicted_patch.append(logits)

        # targets = batch["labels"]  # batch['targets']
        # targets = torch.stack([parse_string_to_tensor(s) for s in targets])
        # max_len = max(target.size(0) for target in targets)
        # targets = torch.stack([torch.cat([target, torch.zeros(max_len - target.size(0))]) for target in targets])
        # targets = targets.clone().detach().to(CFG.device)

        ground_truth_labels_patch.append(targets)

    return loss_meter, predicted_patch, ground_truth_labels_patch


def main():
    # train_df, eval_df = make_train_eval_dfs()

    train_df = make_train_dfs(CFG.captions_path_train)
    eval_df = make_train_dfs(CFG.captions_path_eval)

    print(f"Hallo : train {train_df.shape[0]} valid {eval_df.shape[0]}")

    ###################

    # [:100]
    train_loader = build_loaders(train_df, mode="train")
    valid_loader = build_loaders(eval_df, mode="valid", is_eval=True)

    # all_labels_feature=all_labels_feature,
    # model = CLIPModel(encoded_labels=encoded_labels).to(CFG.device)
    encoded_labels = get_encoded_label(CFG.all_label)
    model = CLIPModel(encoded_labels=encoded_labels).to(CFG.device)
    # model = CLIPModel(encoded_labels=encoded_labels).to(CFG.device)
    # model.load_state_dict(torch.load("results/Coco-MuMicSLS-ALpha/model-epoch-6.pt", map_location=CFG.device))

    print("model is loaded")

    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
    ]

    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )

    step = "epoch"

    # if CFG.debug is not True:
    result = ResultReport()

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)

        model.eval()
        with torch.no_grad():
            valid_loss, predicted_patch, ground_truth_labels_patch = valid_epoch(model, valid_loader)

        (macro_mAP, weighted_mAP, precision, recall, f1, precision_k_1,
         recall_k_1, f1_k_1, precision_k_3, recall_k_3, f1_k_3,
         precision_k_5, recall_k_5, f1_k_5, precision_k_10, recall_k_10, f1_k_10) = Evaluation.evaluate(
            predicted_patch,
            ground_truth_labels_patch)

        if CFG.debug is not True:
            model_path = result.set_epoch_model_name(epoch)
            data = [epoch + 1,
                    round(train_loss,3),
                    round(valid_loss.avg,3),
                    round(macro_mAP.item(), 3),
                    round(weighted_mAP.item(), 3),
                    round(precision, 3),
                    round(recall, 3),
                    round(f1, 3),
                    round(precision_k_1, 3),
                    round(recall_k_1, 3),
                    round(f1_k_1, 3),
                    round(precision_k_3, 3),
                    round(recall_k_3, 3),
                    round(f1_k_3, 3),
                    round(precision_k_5, 3),
                    round(recall_k_5, 3),
                    round(f1_k_5, 3),
                    round(precision_k_10, 3),
                    round(recall_k_10, 3),
                    round(f1_k_10, 3),
                    ]
            result.set_result(data)
            torch.save(model.state_dict(), model_path)
            print(f"save results: {model_path}")

        lr_scheduler.step(valid_loss.avg)



if __name__ == '__main__':
    print(
        f"model run on device {CFG.device} and the temperature is {CFG.temperature} with image encoder {CFG.image_encoder_model}")
    main()
