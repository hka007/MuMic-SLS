from torch.nn import BCEWithLogitsLoss
from ImageEncoder import ImageEncoder
from TextEncoder import TextEncoder
import torch.nn.functional as F
from config import CFG
import torch.nn as nn
import torch

def parse_string_to_tensor(s):
    # Remove brackets and split the string by comma
    s = s.strip('[]').split(',')
    # Convert the string elements to float and create a tensor
    tensor = torch.tensor([float(x) for x in s], dtype=torch.float32)
    return tensor


def calculate_loss(logits, targets, pos_weights=None):
    targets = targets.to(CFG.device)
    logits = logits.to(CFG.device)
    # pos_weights = pos_weights.to(CFG.device)
    criterion = BCEWithLogitsLoss()  # weight=pos_weights
    loss = criterion(logits, targets)

    # criterion = TemperedSigmoidBCELoss()  # weight=pos_weights
    # loss = criterion(logits, targets)
    # criterion = AsymmetricBCEWithLogitsLoss()
    # loss = criterion(logits, targets)

    return loss


def selective_language_supervision(batch_labels, alpha=0.7):
    batch_size, num_classes = batch_labels.size()
    pos_in_batch = []
    for i in range(batch_size):
        Spos = (batch_labels[i] == 1).nonzero(as_tuple=True)[0].tolist()
        pos_in_batch.append(Spos)

    all_pos_in_batch = torch.tensor(list(set(item for sublist in pos_in_batch for item in sublist)))
    all_neg_in_batch = torch.tensor([idx for idx in range(num_classes) if idx not in all_pos_in_batch])

    Sslt = min(alpha * len(all_pos_in_batch), num_classes - len(all_pos_in_batch))

    k = round(Sslt)

    # print("all_pos_in_batch: ", alpha * len(all_pos_in_batch))
    # print("k: ", num_classes - len(all_pos_in_batch))

    random_select_all_neg_indices = torch.randperm(len(all_neg_in_batch))[:k]
    random_select_all_neg = all_neg_in_batch[random_select_all_neg_indices]

    batch_new_label_indices = torch.concat([all_pos_in_batch, random_select_all_neg])
    new_batch_labels = []

    # print(num_classes)
    # print(len(batch_new_label_indices))

    for i in range(batch_size):
        new_label = batch_labels[i, batch_new_label_indices]
        new_batch_labels.append(new_label)

    return torch.stack(new_batch_labels), batch_new_label_indices


class CLIPModel(nn.Module):
    def __init__(
            self,
            temperature=CFG.temperature,
            image_embedding=CFG.image_embedding,
            text_embedding=CFG.text_embedding,
            encoded_labels=None,
            pos_weights=None

    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.encoded_labels = encoded_labels

        self.image_embedding = nn.Linear(image_embedding, CFG.projection_dim)  # Add an embedding layer
        self.text_embedding = nn.Linear(text_embedding, CFG.projection_dim)  # Add an embedding layer

        self.logit_scale = nn.Parameter(torch.tensor([torch.log(torch.tensor(1.0 / temperature))]))

        self.multihead_attn = nn.MultiheadAttention(CFG.projection_dim, num_heads=8, dropout=0, device=CFG.device)

        self.W_i = nn.Parameter(torch.randn(image_embedding, CFG.projection_dim))  # [n, d_e] [32,256]
        self.W_l = nn.Parameter(torch.randn(text_embedding, CFG.projection_dim))  # [n, d_e] [32,256]
        self.W_t = nn.Parameter(torch.randn(text_embedding, CFG.projection_dim))  # [n_c, d_e] [80,256]

        self.pos_weights = pos_weights

    def forward(self, batch, encoded_labels):
        I = batch['image']

        # extract feature representations of each modality
        image_features = self.image_encoder(I)  # [n, d_i] [32, 2048]

        if self.training:
            targets = torch.stack([parse_string_to_tensor(s) for s in batch["labels"]])
            targets = targets.clone().detach().to(CFG.device)
            targets, indices = selective_language_supervision(targets)
            input_ids = encoded_labels['input_ids'][indices]
            attention_mask = encoded_labels['attention_mask'][indices]

        else:
            targets = torch.stack([parse_string_to_tensor(s) for s in batch["labels"]])
            targets = targets.clone().detach().to(CFG.device)
            input_ids = encoded_labels['input_ids']
            attention_mask = encoded_labels['attention_mask']


        all_labels_feature = self.text_encoder(
            input_ids=(input_ids).to(device=CFG.device),
            attention_mask=(attention_mask).to(device=CFG.device)
        )

        image_embeddings = F.normalize(torch.matmul(image_features, self.W_i), p=2,
                                       dim=1)  # L2 normalization along dim=1
        all_labels_embeddings = F.normalize(torch.matmul(all_labels_feature, self.W_t), p=2,
                                            dim=1)  # L2 normalization along dim=1


        # if self.training:
        #     targets = torch.stack([parse_string_to_tensor(s) for s in batch["labels"]])
        #     targets = targets.clone().detach().to(CFG.device)
        #     targets, indices = selective_language_supervision(targets)
        #     all_labels_embeddings = all_labels_embeddings[indices]
        #
        # else:
        #     targets = torch.stack([parse_string_to_tensor(s) for s in batch["labels"]])
        #     targets = targets.clone().detach().to(CFG.device)

        expo = torch.exp(self.logit_scale)

        logits = ((image_embeddings @ all_labels_embeddings.T) * expo).to(
            CFG.device)  # * torch.exp(self.logit_scale)

        loss = calculate_loss(logits, targets)

        return loss, logits, targets

    def predict(self, data):
        I = data['image']
        image_features = self.image_encoder(I)  # [n, d_i] [32, 2048]

        all_labels_feature = self.text_encoder(
            input_ids=(self.encoded_labels['input_ids']).to(device=CFG.device),
            attention_mask=(self.encoded_labels['attention_mask']).to(device=CFG.device))

        # Joint Multimodal Embedding
        image_embeddings = F.normalize(torch.matmul(image_features, self.W_i), p=2,
                                       dim=1)  # L2 normalization along dim=1
        all_labels_embeddings = F.normalize(torch.matmul(all_labels_feature, self.W_t), p=2,
                                            dim=1)  # L2 normalization along dim=1
        expo = torch.exp(self.logit_scale)
        print(expo)

        logits = ((image_embeddings @ all_labels_embeddings.T) * expo).to(
            CFG.device)  # * torch.exp(self.logit_scale)

        return logits
