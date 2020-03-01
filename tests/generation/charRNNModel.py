import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('../../pharmatorch')
from datasets.mosesdata import MosesDataset
from models.generation.charRNN import charRNN
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.nn.utils.rnn import pad_sequence


class charRNNModel(pl.LightningModule):
    """
    Lightning Module for training CharRNN

    """
    @classmethod
    def help(cls):
        print("Hyper parameters include the following \n 1) Dataset path \n 2) Config parameters is a dictionary for CharRNN [hidden_nodes,num_layers,dropout,type_of_RNN_layer]")

    def __init__(self,hparams):
        super(charRNNModel,self).__init__()
        self.root = hparams.root
        self.batch_size = hparams.batch_size
        self.lr = hparams.lr
        self.trainset = MosesDataset(self.root,split="train")
        self.testset = MosesDataset(self.root,split="test_scaffolds")
        self.config = {'hidden_nodes': hparams.hidden_nodes, 'num_layers': hparams.num_layers,'dropout': hparams.dropout, 'rnn': hparams.rnn, 'pad': self.trainset.c2i['<pad>']}
        self.net = charRNN(self.trainset.vocab,self.config)
    def forward(self, x,lens):
        return self.net(x,lens)
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt] ,[sch]
    def train_dataloader(self):
        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = [self.trainset.string2tensor(string) for string in data]
            prevs = pad_sequence([t[:-1] for t in tensors], batch_first=True, padding_value=self.trainset.c2i['<pad>'])
            nexts = pad_sequence([t[1:] for t in tensors], batch_first=True, padding_value=self.trainset.c2i['<pad>'])
            lens = torch.tensor([len(t) - 1 for t in tensors])
            return prevs, nexts, lens
        return DataLoader(self.trainset, collate_fn=collate, batch_size=self.batch_size, shuffle=True)
    def test_dataloader(self):
        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = [self.testset.string2tensor(string) for string in data]
            prevs = pad_sequence([t[:-1] for t in tensors], batch_first=True, padding_value=self.testset.c2i['<pad>'])
            nexts = pad_sequence([t[1:] for t in tensors], batch_first=True, padding_value=self.testset.c2i['<pad>'])
            lens = torch.tensor([len(t) - 1 for t in tensors])
            return prevs, nexts, lens
        return DataLoader(self.testset, collate_fn=collate, batch_size=self.batch_size, shuffle=False)
    def training_step(self, batch,batch_nb):
        prevs, nexts, lens = batch
        out, out_len, _ = self.forward(prevs,lens)
        loss = F.cross_entropy(out.view(-1, out.shape[-1]),nexts.view(-1))
        return {'loss': loss}
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/kollis/pharmatorch/pharmatorch/data/dataset_v2.csv",help="path where dataset is stored")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--hidden_nodes", type=int, default=64, help="adam: learning rate")
    parser.add_argument("--num_layers", type=int, default=2, help="adam: learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="adam: learning rate")
    parser.add_argument("--rnn", type=str, default="LSTM", help="adam: learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="adam: learning rate")
    hparams = parser.parse_args()
    model = charRNNModel(hparams)
    trainer = pl.Trainer()
    trainer.fit(model)

