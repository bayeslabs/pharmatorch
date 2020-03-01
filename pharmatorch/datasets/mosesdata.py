from torch.utils.data import Dataset
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence


class MosesDataset(Dataset):
    """

    Using Moses Benchmarking data for molecular generation
    datasetlink : ' https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv '

    """
    def __init__(self,root_path,split):
        self.root = root_path
        self.data = pd.read_csv(self.root)
        self.data = self.data[self.data["SPLIT"]==split]
        self.data = self.data["SMILES"].squeeze().tolist()
        self.chars = set()
        for string in self.data:
            self.chars.update(string)
        all_sys = sorted(list(self.chars)) + ['<bos>', '<eos>', '<pad>', '<unk>']
        self.c2i = {c: i for i, c in enumerate(all_sys)}
        self.i2c = {i: c for i, c in enumerate(all_sys)}
        self.vocab = all_sys

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    def char2id(self,char):
        if char not in self.c2i:
            return self.c2i['<unk>']
        else:
            return self.c2i[char]

    def id2char(self,id):
        if id not in self.i2c:
            return self.i2c[32]
        else:
            return self.i2c[id]

    def string2ids(self,string, add_bos=False, add_eos=False):
        ids = [self.char2id(c) for c in string]
        if add_bos:
            ids = [self.c2i['<bos>']] + ids
        if add_eos:
            ids = ids + [self.c2i['<eos>']]
        return ids

    def ids2string(self,ids, rem_bos=True, rem_eos=True):
        if len(ids) == 0:
            return ''
        if rem_bos and ids[0] == self.c2i['<bos>']:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.c2i['<eos>']:
            ids = ids[:-1]
        string = ''.join([self.id2char(id) for id in ids])
        return string

    def string2tensor(self, string):
        ids = self.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(ids, dtype=torch.long)
        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.ids2string(ids, rem_bos=True, rem_eos=True)
        return string
