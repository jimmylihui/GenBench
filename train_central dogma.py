import argparse

import numpy as np
import pandas as pd
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC


class DNADataset(Dataset):
    def __init__(self, x_a: np.ndarray, x_c: np.ndarray, labels: np.ndarray, tokenizer):
        super().__init__()

        self.x_a = x_a
        self.x_c = x_c
        self.labels = labels

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.x_a)

    def __getitem__(self, idx):
        x_a = self.tokenizer(
            self.x_a[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        )
        x_a = {k: v.squeeze(0) for k, v in x_a.items()}
        x_c = self.tokenizer(
            self.x_c[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,
        )
        x_c = {k: v.squeeze(0) for k, v in x_c.items()}

        return x_a, x_c, self.labels[idx]


class DataModule(L.LightningDataModule):
    def __init__(self, batch_size: int, tokenizer):
        super().__init__()

        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def setup(self, stage: str) -> None:
        data = pd.read_csv("./updated_sequences_test.csv")
        x_a = data["seq_a"].to_numpy()
        x_c = data["seq_c"].to_numpy()
        labels = data["label"].to_numpy()

        perm = np.random.permutation(len(data))
        x_a = x_a[perm]
        x_c = x_c[perm]
        labels = labels[perm]

        train_size = int(0.8 * len(data))
        train_x_a = x_a[:train_size]
        train_x_c = x_c[:train_size]
        train_labels = labels[:train_size]

        val_x_a = x_a[train_size:]
        val_x_c = x_c[train_size:]
        val_labels = labels[train_size:]

        self.train_dataset = DNADataset(train_x_a, train_x_c, train_labels, self.tokenizer)
        self.val_dataset = DNADataset(val_x_a, val_x_c, val_labels, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=True,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
        )


class SeqClassifier(L.LightningModule):
    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder

        if hasattr(self.encoder.config, "d_model"):
            hidden_size = self.encoder.config.d_model
        elif hasattr(self.encoder.config, "hidden_size"):
            hidden_size = self.encoder.config.hidden_size
        else:
            raise ValueError("Unknown hidden size")

        self.classifier = nn.Linear(hidden_size * 2, 2)

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()

        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x_a, x_c) -> torch.Tensor:
        with torch.no_grad():
            self.encoder.eval()
            hidden_states_a = self.encoder(**x_a)[0]
            hidden_states_c = self.encoder(**x_c)[0]

            # 获取平均嵌入
            embedding_a = torch.mean(hidden_states_a, dim=1)
            embedding_c = torch.mean(hidden_states_c, dim=1)

            # 连接嵌入
            combined_embeddings = torch.cat((embedding_a, embedding_c), dim=1)

        # 通过线性层进行分类
        logits = self.classifier(combined_embeddings)
        return logits

    def training_step(self, batch, batch_idx):
        x_a, x_c, labels = batch
        logits = self(x_a, x_c)
        loss = F.cross_entropy(logits, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        self.train_auroc(preds, labels)

        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), on_epoch=True, prog_bar=True)
        self.log("train_auroc", self.train_auroc.compute(), on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x_a, x_c, labels = batch
        logits = self(x_a, x_c)
        loss = F.cross_entropy(logits, labels)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)
        self.val_auroc(preds, labels)

        return loss
    
    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), on_epoch=True, prog_bar=True)
        self.log("val_auroc", self.val_auroc.compute(), on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.classifier.parameters(), lr=1e-4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="DNABERT")
    args = parser.parse_args()

    model_name = args.model_name

    if model_name == "DNABERT":
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True)
        encoder = AutoModel.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True)
    elif model_name == "DNABERT2":
        tokenizer = AutoTokenizer.from_pretrained("vivym/DNABERT-2-117M", trust_remote_code=True)
        encoder = AutoModel.from_pretrained("vivym/DNABERT-2-117M", trust_remote_code=True)
    elif model_name == "GENA_LM":
        tokenizer = AutoTokenizer.from_pretrained("AIRI-Institute/gena-lm-bert-base-t2t", trust_remote_code=True)
        encoder = AutoModel.from_pretrained("AIRI-Institute/gena-lm-bert-base-t2t", trust_remote_code=True)
        encoder = encoder.bert
    elif model_name == "Caduceus":
        tokenizer = AutoTokenizer.from_pretrained("kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16", trust_remote_code=True)
        encoder = AutoModel.from_pretrained("kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16", trust_remote_code=True)
    elif model_name == "HyenaDNA":
        tokenizer = AutoTokenizer.from_pretrained("LongSafari/hyenadna-medium-160k-seqlen-hf", trust_remote_code=True)
        encoder = AutoModel.from_pretrained("LongSafari/hyenadna-medium-160k-seqlen-hf", trust_remote_code=True)
    elif model_name == "NT":
        tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", trust_remote_code=True)
        encoder = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", trust_remote_code=True).esm
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    trainer = L.Trainer(
        max_epochs=50,
        devices=1,
        callbacks=[
            ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                save_top_k=5,
                save_weights_only=True,
                filename='{epoch}-{val_acc:.2f}',
            ),
        ],
        logger=TensorBoardLogger("logs", name=model_name),
        precision=16,
    )

    dm = DataModule(batch_size=256, tokenizer=tokenizer)

    model = SeqClassifier(encoder)

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
