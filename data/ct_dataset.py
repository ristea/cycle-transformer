# This code is released under the CC BY-SA 4.0 license.

import pickle

import numpy as np
import pydicom

from data.base_dataset import BaseDataset


class CTDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.raw_data = pickle.load(open(opt.dataroot, "rb"))
        self.labels = None
        self.Aclass = opt.Aclass
        self.Bclass = opt.Bclass
        self._make_dataset()

    def _make_dataset(self):
        data = []

        for entity in self.raw_data:
            if entity in self.Aclass:
                data += self.raw_data[entity]
        self.raw_data = data

    def __getitem__(self, index):
        # Image from A
        A_image = pydicom.dcmread(self.raw_data[index]).pixel_array
        A_image[A_image < 0] = 0
        A_image = A_image / 1e3
        A_image = A_image - 1
        A_image = np.expand_dims(A_image, 0).astype(np.float)

        # Paired image from B
        path = self.raw_data[index].replace(self.Aclass, self.Bclass)
        B_image = pydicom.dcmread(path).pixel_array
        B_image[B_image < 0] = 0
        B_image = B_image / 1e3
        B_image = B_image - 1
        B_image = np.expand_dims(B_image, 0).astype(np.float)

        return {'A': A_image, 'B': B_image}

    def __len__(self):
        return len(self.raw_data)
