import torch
import pandas as pd
from torchmetrics.classification import MultilabelAveragePrecision
from tqdm import tqdm
from config import CFG
from Train import build_loaders
from torchmetrics import AveragePrecision
from transformers import DistilBertTokenizer
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, classification_report

from MuMicSLS import CLIPModel


def mAP_k(y_targets, y_prediction, k=5):
    y_prediction = torch.tensor(y_prediction, dtype=torch.float32, device=CFG.device)
    y_targets = torch.tensor(y_targets, dtype=torch.float32, device=CFG.device)
    # Get the top-k indices of y_prediction
    _, top_indices = y_prediction.topk(k, dim=1)

    # Gather the corresponding target values using the top-k indices
    topk_targets = y_targets.gather(1, top_indices).to(CFG.device)

    cumsum = torch.cumsum(topk_targets, dim=1).to(CFG.device)
    positions = torch.arange(1, k + 1).float().unsqueeze(0).to(CFG.device)
    precision_at_k = (cumsum / positions).to(CFG.device)

    # Calculate the mean of the precision values for each example
    AP_at_k = precision_at_k.mean(dim=1)
    mAP_at_k = AP_at_k.mean().item()

    return mAP_at_k


def parse_string_to_tensor(s):
    # Remove brackets and split the string by comma
    s = s.strip('[]').split(',')
    # Convert the string elements to float and create a tensor
    tensor = torch.tensor([float(x) for x in s], dtype=torch.float32)
    return tensor


def calculate_metrics_at_k(true_labels, scores, k=5):
    batch_size = true_labels.size(0)

    # Container for the metrics
    precisions = []
    recalls = []
    f1_scores = []

    top_values, top_indices = torch.topk(scores, k)

    for i in range(batch_size):
        y_true = true_labels[i]
        y_scores = scores[i]

        # Get indices of the top k scores
        _, top_indices = torch.topk(y_scores, k)

        # Get the true labels for these indices
        true_labels_at_k = y_true[top_indices]

        # Assume all k predictions are positives
        predictions_at_k = torch.ones(k, dtype=torch.int)

        # Calculate metrics for the current example
        precision = precision_score(true_labels_at_k.cpu(), predictions_at_k.cpu(), zero_division=0)
        recall = recall_score(true_labels_at_k.cpu(), predictions_at_k.cpu(), zero_division=0)
        f1 = f1_score(true_labels_at_k.cpu(), predictions_at_k.cpu(), zero_division=0)

        # Append metrics for this example
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Average the precision, recall, and F1 score across the batch
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    print(f"Precision@{k}: {avg_precision:.2f}")
    print(f"Recall@{k}: {avg_recall:.7f}")
    print(f"F1@{k}: {avg_f1:.2f}")

    return avg_precision, avg_recall, avg_f1


def evaluate_by_model(model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)

    # all_labels_df = pd.read_csv("Test_set/test_all_labels_rubric_aspect_element.csv")
    all_labels_df = pd.read_csv(CFG.all_label)  # testalllabels
    all_labels = all_labels_df.values
    flattened_list = [item for sublist in all_labels for item in sublist]
    flattened_list = [cap.lower() for cap in flattened_list]

    flattened_list = [f"this image contains {word.replace(',', ' and ')}" for word in flattened_list]

    number_of_labels = len(flattened_list)
    encoded_labels = tokenizer(flattened_list, padding=True, truncation=True, return_tensors='pt',
                               max_length=CFG.max_length)

    model = CLIPModel(encoded_labels=encoded_labels).to(CFG.device)

    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    print(model_path)
    model.eval()

    test_df = pd.read_csv(CFG.captions_path)  #
    test_loader = build_loaders(test_df[:5000], tokenizer, is_eval=True)

    predicted_patch = []
    ground_truth_labels_patch = []

    tqdm_object = tqdm(test_loader, total=len(test_loader))
    for batch in tqdm_object:
        # Forward pass through the model to get the loss (you may modify this if needed)
        batch = {k: (v.to(CFG.device) if k not in ["caption", "labels"] else v) for k, v in batch.items()}

        with torch.no_grad():  # Disable gradient calculation for inference
            loss, logits, targets = model(batch, encoded_labels=encoded_labels)  #
            logits = torch.sigmoid(logits).to(CFG.device)

            predicted_patch.append(logits)

            targets = batch["labels"]  # batch['targets']
            targets = torch.stack([parse_string_to_tensor(s) for s in targets])
            max_len = max(target.size(0) for target in targets)
            targets = torch.stack([torch.cat([target, torch.zeros(max_len - target.size(0))]) for target in targets])
            targets = targets.clone().detach().to(CFG.device)

            ground_truth_labels_patch.append(targets)

    max_len = max(pred.size(0) for pred in predicted_patch)
    predicted_scores = torch.stack(
        [torch.cat([pred.to(CFG.device),
                    torch.zeros(max_len - pred.size(0), number_of_labels, device=CFG.device).to(CFG.device)]) for pred
         in predicted_patch])
    max_len = max(pred.size(0) for pred in ground_truth_labels_patch)
    ground_truth_labels = torch.stack(
        [torch.cat([pred.to(CFG.device), torch.zeros(max_len - pred.size(0), number_of_labels, device=CFG.device)]) for
         pred in
         ground_truth_labels_patch])

    targets = torch.tensor(ground_truth_labels, dtype=torch.int, device=CFG.device)
    preds = predicted_scores.to(CFG.device)

    pred = torch.where(torch.isnan(preds[0]), torch.tensor(0.0).to(CFG.device), preds[0].to(CFG.device))
    target = torch.where(torch.isnan(targets[0]), torch.tensor(0, dtype=torch.int64).to(CFG.device),
                         targets[0].to(torch.int64).to(CFG.device))

    macro_average_precision = AveragePrecision(task="multilabel", num_labels=number_of_labels, average="macro").to(
        CFG.device)
    weighted_average_precision = AveragePrecision(task="multilabel", num_labels=number_of_labels,
                                                  average="weighted").to(CFG.device)

    macro_mAP = macro_average_precision(pred, target)
    weighted_mAP = weighted_average_precision(pred, target)

    threshold = 0.5
    k = 10
    y_pred = torch.tensor(pred > threshold, dtype=torch.int, device=CFG.device)
    precision_1, recall_1, f1_1 = calculate_metrics_at_k(target, y_pred, k=1)
    precision_2, recall_2, f1_2 = calculate_metrics_at_k(target, y_pred, k=2)
    precision_5, recall_5, f1_5 = calculate_metrics_at_k(target, y_pred, k=5)
    precision_10, recall_10, f1_10 = calculate_metrics_at_k(target, y_pred, k=10)
    precision_15, recall_15, f1_15 = calculate_metrics_at_k(target, y_pred, k=15)
    precision_20, recall_20, f1_20 = calculate_metrics_at_k(target, y_pred, k=20)

    # Now, y_pred contains the binary class labels
    y_true_np = target.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    precision_micro = precision_score(y_true_np, y_pred_np, average='micro')
    recall_micro = recall_score(y_true_np, y_pred_np, average='micro')
    f1_micro = f1_score(y_true_np, y_pred_np, average='micro')

    print(f"macro_mAP: {macro_mAP}")
    print(f"weighted_mAP: {weighted_mAP}")
    # print(f"precision: {precision}")
    # print(f"recall_macro: {recall_macro}")
    # print(f"recall_micro: {recall_micro}")
    # print(f"recall_weighted: {recall_weighted}")
    # print(f"recall: {recall}")
    # print(f"f1: {f1}")

    # return macro_mAP.item, weighted_mAP.item, precision, recall, f1, recall_macro, recall_micro, recall_weighted, recall_samples, precision_macro, precision_micro, precision_weighted, precision_samples
    return (
        precision_micro, recall_micro, f1_micro,
        precision_1, recall_1, f1_1,
        precision_2, recall_2, f1_2,
        precision_5, recall_5, f1_5,
        precision_10, recall_10, f1_10,
        precision_15, recall_15, f1_15,
        precision_20, recall_20, f1_20)


def evaluate(predicted_patch, ground_truth_labels_patch):
    all_labels_df = pd.read_csv(CFG.all_label)  # all_label
    all_labels = all_labels_df.values
    flattened_list = [item for sublist in all_labels for item in sublist]

    number_of_labels = len(flattened_list)

    max_len = max(pred.size(0) for pred in predicted_patch)
    predicted_scores = torch.stack(
        [torch.cat([pred.to(CFG.device),
                    torch.zeros(max_len - pred.size(0), number_of_labels, device=CFG.device).to(CFG.device)]) for pred
         in predicted_patch])
    max_len = max(pred.size(0) for pred in ground_truth_labels_patch)
    ground_truth_labels = torch.stack(
        [torch.cat([pred.to(CFG.device), torch.zeros(max_len - pred.size(0), number_of_labels, device=CFG.device)]) for
         pred in
         ground_truth_labels_patch])

    targets = torch.tensor(ground_truth_labels, dtype=torch.int, device=CFG.device)
    preds = predicted_scores.to(CFG.device)

    pred = torch.where(torch.isnan(preds[0]), torch.tensor(0.0).to(CFG.device), preds[0].to(CFG.device))
    target = torch.where(torch.isnan(targets[0]), torch.tensor(0, dtype=torch.int64).to(CFG.device),
                         targets[0].to(torch.int64).to(CFG.device))

    mAP_k(target, pred)

    macro_average_precision = AveragePrecision(task="multilabel", num_labels=number_of_labels, average="macro").to(
        CFG.device)
    weighted_average_precision = AveragePrecision(task="multilabel", num_labels=number_of_labels,
                                                  average="weighted").to(CFG.device)

    macro_mAP = macro_average_precision(pred, target)
    weighted_mAP = weighted_average_precision(pred, target)

    threshold = 0.5
    k = 10
    y_pred = torch.tensor(pred > threshold, dtype=torch.int, device=CFG.device)

    precision_k_1, recall_k_1, f1_k_1 = calculate_metrics_at_k(target, y_pred, k=1)
    precision_k_3, recall_k_3, f1_k_3 = calculate_metrics_at_k(target, y_pred, k=3)
    precision_k_5, recall_k_5, f1_k_5 = calculate_metrics_at_k(target, y_pred, k=5)
    precision_k_10, recall_k_10, f1_k_10 = calculate_metrics_at_k(target, y_pred, k=10)

    # Now, y_pred contains the binary class labels
    y_true_np = target.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    precision = precision_score(y_true_np, y_pred_np, average='macro')
    recall_macro = recall_score(y_true_np, y_pred_np, average='macro')
    recall_micro = recall_score(y_true_np, y_pred_np, average='micro')
    recall_weighted = recall_score(y_true_np, y_pred_np, average='weighted')
    recall = recall_score(y_true_np, y_pred_np, average='macro')
    f1 = f1_score(y_true_np, y_pred_np, average='macro')

    maP_weighted = MultilabelAveragePrecision(num_labels=80, average="none", thresholds=5).to(CFG.device)
    maP_micro = MultilabelAveragePrecision(num_labels=80, average="micro", thresholds=5).to(CFG.device)
    maP_macro = MultilabelAveragePrecision(num_labels=80, average="macro", thresholds=5).to(CFG.device)

    return macro_mAP, weighted_mAP, precision, recall, f1, precision_k_1, recall_k_1, f1_k_1, precision_k_3, recall_k_3, f1_k_3, precision_k_5, recall_k_5, f1_k_5, precision_k_10, recall_k_10, f1_k_10


if __name__ == '__main__':
    pass
    print(evaluate_by_model("results/100Labels-Advision-MuMicSLS-allData/model-epoch-4.pt"))
    # df = pd.DataFrame(columns=["macro_mAP", "weighted_mAP", "recall_micro", "precision_micro"])
    df = pd.DataFrame(columns=[
        "precision_micro", "recall_micro", "f1_micro",
        "precision_1", "recall_1", "f1_1",
        "precision_2", "recall_2", "f1_2",
        "precision_5", "recall_5", "f1_5",
        "precision_10", "recall_10", "f1_10",
        "precision_15", "recall_15", "f1_15",
        "precision_20", "recall_20", "f1_20"])

    # results = []
    # for i in range(1):
    #     # print(f"results/sameDATASET-COCO-MuMic-SLS-TO_BE_SURE/model-epoch-{i}.pt")
    #     result = evaluate_by_model(
    #         f"results/sameDATASET-COCO-MuMic-SLS-TO_BE_SURE/model-epoch-{10}.pt")
    #
    #     df = df.append(pd.Series(result, index=df.columns), ignore_index=True)

    print(df)

    df = df.round(3)

    df.to_csv("MuMic_SLS_COCO.csv", index=False)
