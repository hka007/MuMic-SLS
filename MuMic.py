from torch.nn import BCEWithLogitsLoss

from ImageEncoder import ImageEncoder
from config import CFG

from TextEncoder import TextEncoder
import torch
import torch.nn.functional as F
import torch.nn as nn

'''
# image_encoder - vit
# text_encoder - Text Transformer
# I[n, h, w, c] - minibatch of aligned image
# T[n_c, L] - vector of tokenized label-text (n_c is #classes, l is tokens)
# W_i[d_i, d_e] - learned image proj
# W_t[d_t, d_e] - learned text proj
# targets[n, n_c] - the ground truth
# logit_scale  - the logit_scale param, = ln(1/temperature)
# extract feature representations of each modality
I_f = image encoder(I) #[n, d_i]
T_f = image encoder(T) #[n_c, d_t]

# joint multimodal embedding
I_e = L2_normalize(np.dot(I_f, W_i), axis = 1) # [n, d_e]
T_e = L2_normalize(np.dot(T_f, W_t), axis = 1) # [n_c, d_e]

# scaled pairwise cos similarities [n, n_c]
logits = np.dot(I_e, T_e.T) * np.exp(logit_scale)

# loss - only need image level loss
loss = nn.BCEWithLogitsLoss(logits, targets)
'''


def parse_string_to_tensor(s):
    # Remove brackets and split the string by comma
    s = s.strip('[]').split(',')
    # Convert the string elements to float and create a tensor
    tensor = torch.tensor([float(x) for x in s], dtype=torch.float32)
    return tensor


def calculate_loss(logits, targets):
    targets = torch.stack([parse_string_to_tensor(s) for s in targets])
    targets = targets.clone().detach().to(CFG.device)

    criterion = BCEWithLogitsLoss()
    loss = criterion(logits, targets)

    return loss


class CLIPModel(nn.Module):
    def __init__(
            self,
            temperature=CFG.temperature,
            image_embedding=CFG.image_embedding,
            text_embedding=CFG.text_embedding,
            encoded_labels=None
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.encoded_labels = encoded_labels

        self.logit_scale = nn.Parameter(torch.tensor([torch.log(torch.tensor(1.0 / temperature))]))

        self.W_i = nn.Parameter(torch.randn(image_embedding, CFG.projection_dim))  # [n, d_e] [32,256]
        self.W_t = nn.Parameter(torch.randn(text_embedding, CFG.projection_dim))  # [n_c, d_e] [334,256]

        self.image_embedding = nn.Linear(image_embedding, CFG.projection_dim)  # Add an embedding layer
        self.text_embedding = nn.Linear(text_embedding, CFG.projection_dim)  # Add an embedding layer

    def forward(self, batch, encoded_labels=None):


        I = batch['image']
        image_features = self.image_encoder(I)  # [n, d_i] [32, 2048]
        # start_time = time.time()

        all_labels_feature = self.text_encoder(
            input_ids=(self.encoded_labels['input_ids']).to(device=CFG.device),
            attention_mask=(self.encoded_labels['attention_mask']).to(device=CFG.device))

        # end_time = time.time()
        # execution_time = end_time - start_time
        # print("Execution time: {} seconds".format(execution_time))

        # Joint Multimodal Embedding
        image_embeddings = F.normalize(torch.matmul(image_features, self.W_i), p=2,
                                       dim=1)
        text_embeddings = F.normalize(torch.matmul(all_labels_feature, self.W_t), p=2,
                                      dim=1)
        targets = torch.stack([parse_string_to_tensor(s) for s in batch["labels"]])
        targets = targets.clone().detach().to(CFG.device)

        expo = torch.exp(self.logit_scale)

        logits = ((image_embeddings @ text_embeddings.T) * expo).to(
            CFG.device)  # * torch.exp(self.logit_scale)

        loss = calculate_loss(logits, batch["labels"])


        return loss, logits, targets

    def predict(self, batch):
        I = batch['image']
        image_features = self.image_encoder(I)  # [n, d_i] [32, 2048]

        all_labels_feature = self.text_encoder(
            input_ids=(self.encoded_labels['input_ids']).to(device=CFG.device),
            attention_mask=(self.encoded_labels['attention_mask']).to(device=CFG.device))

        # Joint Multimodal Embedding
        image_embeddings = F.normalize(torch.matmul(image_features, self.W_i), p=2,
                                       dim=1)
        text_embeddings = F.normalize(torch.matmul(all_labels_feature, self.W_t), p=2,
                                      dim=1)

        expo = torch.exp(self.logit_scale)

        logits = ((image_embeddings @ text_embeddings.T) * expo).to(
            CFG.device)  # * torch.exp(self.logit_scale)

        return logits
