import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from .utils import read_data

import logging

class BaseDataModule(LightningDataModule):
    def __init__(self, hparams) -> None:
        super().__init__()

        self.save_hyperparameters(hparams)
        self.trainset, self.valset, self.testset = None, None, None
    
    def convert_dataset(self, dataset):
        return torch.utils.data.TensorDataset(* dataset )

    def setup(self, stage='fit'):

        logging.info('Reading data')

        datasets = {
            name: os.path.join(self.hparams['input_dir'], name) for name in ['train', 'val', 'test']
        }

        if stage=='fit':
            self.trainset, self.valset = [
                [torch.tensor(ds, dtype=torch.float32) for ds in read_data([ os.path.join(datasets[name], f) for f in  os.listdir(datasets[name])], max_etot=self.hparams.get('max_etot'), min_etot=self.hparams.get('min_etot'), prog_bar=True)   ]
                for name in ['train', 'val']
            ]
            self.trainset, self.valset = self.transform(self.trainset), self.transform(self.valset)
        
        if stage=='test':
            self.testset = [torch.tensor(ds, dtype=torch.float32) for ds in read_data([ os.path.join(datasets['test'], f) for f in  os.listdir(datasets['test'])], max_etot=self.hparams.get('max_etot'), min_etot=self.hparams.get('min_etot'), prog_bar=True) ]
            self.testset = self.transform(self.testset)

    def train_dataloader(self):
        return DataLoader(
            self.convert_dataset(self.trainset), 
            batch_size=self.hparams['batch_size'],
            num_workers=int(os.cpu_count() / 4)
            )

    def val_dataloader(self):
        return DataLoader(
            self.convert_dataset(self.valset), 
            batch_size=self.hparams['batch_size'],
            num_workers=int(os.cpu_count() / 4))

    def test_dataloader(self):
        return DataLoader(
            self.convert_dataset(self.testset), 
            batch_size=self.hparams['batch_size'],
            num_workers=int(os.cpu_count() / 4))
    
    def transform(self, dataset):
        return [ds for ds in dataset]

class CartesianDataModule(BaseDataModule):
    def __init__(self, hparams) -> None:
        super().__init__(hparams)

    def transform(self, dataset):
        etot, m, outcome = dataset
        outcome = outcome[:, : self.hparams['n_particle'] ].reshape((outcome.shape[0], -1))
        outcome /= etot
        etot /= 1000.
        m /= 1000.
        return [etot, m, outcome]