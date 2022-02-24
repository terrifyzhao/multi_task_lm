import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


class LoopDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_idx = 0
        self.n_batches = len(self.dataset) // self.batch_size
        if len(self.dataset) % self.batch_size != 0:
            self.n_batches += 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_idx >= self.n_batches:
            self.batch_idx = 0
        cur_data = self.dataset[self.batch_idx * self.batch_size: (self.batch_idx + 1) * self.batch_size]
        self.batch_idx += 1
        return cur_data


class MultiTaskDataLoader:
    def __init__(self, datasets, batch_size_list):
        self.datasets = datasets
        self.batch_size_list = batch_size_list
        self.dataloader = [LoopDataLoader(dataset, batch_size) for dataset, batch_size in
                           zip(datasets, batch_size_list)]

    def __iter__(self):
        return self

    def __next__(self):
        cur_data = [next(dataloader) for dataloader in self.dataloader]
        return cur_data
