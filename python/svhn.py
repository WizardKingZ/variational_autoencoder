"""
The following code is from
https://github.com/ermongroup/Variational-Ladder-Autoencoder/blob/master/dataset/svhn.py
"""
import scipy.io as sio
from python.dataset import Dataset
import numpy as np
import os

class SVHN(Dataset):
    def __init__(self, db_path=''):
        Dataset.__init__(self)
        print("Loading files")
        self.data_dims = [32, 32, 3]
        self.range = [0.0, 1.0]
        self.name = "svhn"
        self.train_file = os.path.join(db_path, "train_32x32.mat")
        self.test_file = os.path.join(db_path, "test_32x32.mat")

        # Load training images
        if os.path.isfile(self.train_file):
            mat = sio.loadmat(self.train_file)
            self.train_image = mat['X'].astype(np.float32)
            self.train_label = mat['y']
            self.train_image = np.clip(self.train_image / 255.0, a_min=0.0, a_max=1.0)
        else:
            print("SVHN dataset train files not found")
            exit(-1)
        self.train_batch_ptr = 0
        self.train_size = self.train_image.shape[-1]

        if os.path.isfile(self.test_file):
            mat = sio.loadmat(self.test_file)
            self.test_image = mat['X'].astype(np.float32)
            self.test_label = mat['y']
            self.test_image = np.clip(self.test_image / 255.0, a_min=0.0, a_max=1.0)
        else:
            print("SVHN dataset test files not found")
            exit(-1)
        self.test_batch_ptr = 0
        self.test_size = self.test_image.shape[-1]
        print("SVHN loaded into memory")

    def next_batch(self, batch_size):
        prev_batch_ptr = self.train_batch_ptr
        self.train_batch_ptr += batch_size
        if self.train_batch_ptr > self.train_image.shape[-1]:       # Note the ordering of dimensions
            self.train_batch_ptr = batch_size
            prev_batch_ptr = 0
        return np.transpose(self.train_image[:, :, :, prev_batch_ptr:self.train_batch_ptr], (3, 0, 1, 2)).reshape([batch_size, -1]), None

    def next_test_batch(self, batch_size):
        prev_batch_ptr = self.test_batch_ptr
        self.test_batch_ptr += batch_size
        if self.test_batch_ptr > self.test_image.shape[-1]:
            self.test_batch_ptr = batch_size
            prev_batch_ptr = 0
        return np.transpose(self.test_image[:, :, :, prev_batch_ptr:self.test_batch_ptr], (3, 0, 1, 2)).reshape([batch_size, -1])

    def display(self, image):
        return np.clip(image, 0.0, 1.0)

    def reset(self):
        self.train_batch_ptr = 0
        self.test_batch_ptr = 0