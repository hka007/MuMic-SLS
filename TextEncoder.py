from torch import nn
from config import CFG
from transformers import DistilBertModel, DistilBertConfig
from transformers import GPT2Model, GPT2Config


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        # Define the classification layer
        # self.classification_layer = nn.Linear(self.model.config.hidden_size, CFG.num_classes)

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state

        # Pass the CLS token representation through the classification layer
        cls_token_rep = last_hidden_state[:, self.target_token_idx, :]

        # return self.classification_layer(cls_token_rep)
        return cls_token_rep


class TextEncoderGPT(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = GPT2Model.from_pretrained(model_name)
        else:
            self.model = GPT2Model(config=GPT2Config())

        for p in self.model.parameters():
            p.requires_grad = trainable

        # GPT-2 uses the last token for representing the entire sequence, unlike BERT which uses the first (CLS) token
        self.target_token_idx = -1

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state

        # In GPT-2, the representation of the entire sequence is typically taken from the last token
        last_token_rep = last_hidden_state[:, self.target_token_idx, :]

        # You can also use a classification layer like before if needed
        # return self.classification_layer(last_token_rep)
        return last_token_rep