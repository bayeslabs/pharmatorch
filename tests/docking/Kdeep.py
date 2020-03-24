import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('../../pharmatorch')
from datasets.dockingdata import DockingDataset
from models.docking.squeezenet import SqueezeNet
import torch.nn.functional as F
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint

class Sqeezenetmodel(pl.LightningModule):

    def __init__(self,hparams):
        super(Sqeezenetmodel,self).__init__()
        self.root = hparams.root
        self.batch_size = hparams.batch_size
        self.lr = hparams.lr
        #self.split = hparams.split
        self.dataset = DockingDataset(self.root)
        self.trainset = self.dataset[:2]
        self.testset = self.dataset[2:]
        self.net = SqueezeNet()
    def forward(self, x):
        return self.net(x)
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt] ,[sch]
    def train_dataloader(self):
        return DataLoader(self.trainset,batch_size=self.batch_size)
    def test_dataloader(self):
        return DataLoader(self.testset,batch_size=self.batch_size)
    def training_step(self, batch, batch_nb):
        x,y = batch[0],batch[1]
        y_hat = self.forward(x)
        loss = F.mse_loss(y,y_hat)
        return {'loss': loss}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/kollis/pharmatorch/data/microdata",
                        help="path where dataset is stored")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    hparams = parser.parse_args()
    model = Sqeezenetmodel(hparams)
    trainer = pl.Trainer()
    trainer.fit(model)

