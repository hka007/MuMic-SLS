import datetime
import os

import pandas as pd

from config import CFG


class ResultReport:
    def __init__(
            self,
            dataset=CFG.dataset,
            image_encoder=CFG.image_encoder_model,
            alpha=CFG.alpha,
            gamma=CFG.gamma,
            beta=CFG.beta,
            temperature=CFG.temperature,

    ):
        super().__init__()
        self.folder_name = f"results/Coco-"

        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

        self.model_name = f"model-epoch-"
        self.result_name = "result.csv"

        self.columns = ["Epoch", "train_loss",
                        "valid_loss", "mAP", "weighted_mAP",
                        "precision", "recall", "f1",
                        "precision@1", "recall@1", "f1@1",
                        "precision@3", "recall@3", "f1@3",
                        "precision@5", "recall@5", "f1@5",
                        "precision@10", "recall@10", "f1@10"]

        self.df_result = pd.DataFrame(columns=self.columns)
        self.data = []

    def set_epoch_model_name(self, epoch):
        now = datetime.datetime.now()

        return self.folder_name + "/" + self.model_name + str(epoch) + ".pt"

    def set_result(self, data):
        self.df_result = pd.concat([self.df_result, pd.DataFrame([data], columns=self.columns)], ignore_index=True)

        result_path = self.folder_name + "/" + self.result_name
        self.df_result.to_csv(result_path, index=False)
