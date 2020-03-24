from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import os
import fnmatch
from moleculekit.molecule import Molecule
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors, viewVoxelFeatures
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.smallmol.smallmol import SmallMol


class DockingDataset(Dataset):

    @classmethod
    def help(cls):
        print("")

    def __init__(self,root_path):
        self.root =root_path
        self.data = pd.read_csv(os.path.join(self.root,"affinity.csv"))
        self.data = [tuple((self.get_voxel(val[0]),val[1])) for val in self.data.values]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

    def get_voxel(self,path):
        complex = os.path.join(self.root,path)
        for ele in os.listdir(complex):
            if fnmatch.fnmatch(ele, '*_protein.pdb'):
                prot = Molecule(os.path.join(complex, ele))
                prot.filter('protein')

                # If your structure is fully protonated and contains all bond information in prot.bonds skip this step!
                prot = prepareProteinForAtomtyping(prot)


                prot.view(guessBonds=False)
                prot_vox, prot_centers, prot_N = getVoxelDescriptors(prot, boxsize=[24, 24, 24], center=[0, 0, 0],
                                                                     buffer=1)
                prot.view(guessBonds=False)
                viewVoxelFeatures(prot_vox, prot_centers, prot_N)

                nchannels = prot_vox.shape[1]

                prot_vox_t = prot_vox.transpose().reshape([1, nchannels, prot_N[0], prot_N[1], prot_N[2]])
                prot_vox_t = torch.tensor(prot_vox_t.astype(np.float32))

                for ele in os.listdir(complex):
                    if fnmatch.fnmatch(ele, '*_ligand.mol2'):
                        slig = SmallMol(os.path.join(os.path.join(complex, ele)),force_reading=True)
                        slig.view(guessBonds=False)

                        # For the ligand since it's small we could increase the voxel resolution if we so desire to 0.5 A instead of the default 1 A.
                        lig_vox, lig_centers, lig_N = getVoxelDescriptors(slig, boxsize=[24, 24, 24], center=[0, 0, 0],
                                                                          voxelsize=1, buffer=1)
                        slig.view(guessBonds=False)
                        viewVoxelFeatures(lig_vox, lig_centers, lig_N)

                        lig_vox_t = lig_vox.transpose().reshape([1, nchannels, lig_N[0], lig_N[1], lig_N[2]])
                        lig_vox_t = torch.tensor(lig_vox_t.astype(np.float32))

        x = torch.cat((lig_vox_t, prot_vox_t), 1)
        x.squeeze_(0)
        return x




