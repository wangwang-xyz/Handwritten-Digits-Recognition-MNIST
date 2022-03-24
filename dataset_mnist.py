import os.path
import struct
import numpy as np

from torch.utils.data import Dataset

class DataMNIST(Dataset):

    def __init__(self, rootdir, dataname, labelname, transform = None):
        self.data = load_training_data(os.path.join(rootdir, dataname))
        self.label = load_lable_data(os.path.join(rootdir, labelname))
        self.transform = transform

    def __getitem__(self, idx):
        img = self.data[idx]
        # img = norm(img)
        img = img.reshape(28, 28)
        img = np.array(img, dtype='uint8')

        if self.transform != None:
            img = self.transform(img)
        target = int(self.label[idx])
        return img, target

    def __len__(self):
        return len(self.label)

def norm(x):
    num = x.size
    MAX = np.max(x)
    MIN = np.min(x)
    x = x / (MAX - MIN)
    MAX = np.max(x)
    MIN = np.min(x)
    avr = np.sum(x) / num
    x = (x - avr) / (MAX - MIN)

    return x

def load_training_data(address):
    with open(address, 'rb') as f:
        temp = f.read()

    head = struct.unpack_from('>IIII', temp, 0)

    offset = struct.calcsize('>IIII')

    num = head[1]
    width = head[2]
    height = head[3]

    x = width * height
    y = num

    bits = x * y
    bitsString = ('>' + str(bits) + 'B')
    image = struct.unpack_from(bitsString, temp, offset)
    f.close()
    data = (np.reshape(image, (x, y), order='F') * 1.)


    return data.T


def load_lable_data(address):
    with open(address, 'rb') as f:
        temp = f.read()

    head = struct.unpack_from('>II', temp, 0)

    offset = struct.calcsize('>II')

    num = head[1]

    bits = num
    bitsString = ('>' + str(bits) + 'B')
    lable = struct.unpack_from(bitsString, temp, offset)
    f.close()
    Y = np.reshape(lable, (1, num))

    return Y.T