from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class ECGDataModule(L.LightningDataModule):
    def __init__(self, processed_path: str, batch_size: int, val_split: float = 0.2):
        super().__init__()
        self.processed_path = Path(processed_path)
        self.batch_size = batch_size
        self.val_split = val_split

    def setup(self, stage=None):
        # Загружаем сохраненные в preprocess.py тензоры
        data = torch.load(self.processed_path)
        dataset = TensorDataset(data["x"], data["y"])

        # Разбиваем на train и val
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size

        self.train_ds, self.val_ds = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=0)
