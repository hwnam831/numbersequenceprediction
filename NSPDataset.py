import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

'''
# 16-dim one-hot vectors
# 0-9:  decimal digits
# 10:   Delimiter (space)
# 11:   Pad
# 12:   Start
# 13:   EOS
# 14:   Mask
# 15:   Custom token    
'''

class Token():
    delim = 10 
    pad = 11
    start = 12
    eos = 13
    mask = 14
    custom = 15

def fib(seed1, seed2, numbers):
    seq = [seed1, seed2]
    for i in range(2, numbers):
        seq.append(seq[i-2]+seq[i-1])
    target = seq[-2]+seq[-1]
    return seq, target

def num2vec(num, ndigits, lendian=True):
    digits = [int(c) for c in str(num)]
    if len(digits)<ndigits:
        digits = [0 for i in range(ndigits-len(digits))]+digits
    elif len(digits)>ndigits:
        digits = digits[len(digits)-ndigits:]
    if lendian:
        digits.reverse()
    return np.array(digits)

class NSPDataset(Dataset):
    def __init__(self, rule, maxdigits, mindigits=1, numbers=2, size=25600, lendian=True):
        self.rule = rule
        assert maxdigits > mindigits
        self.maxdigits = maxdigits
        self.mindigits = mindigits
        self.size = size
        self.lendian = lendian
        self.numbers = numbers
        self.inputlen = (maxdigits+1)*numbers + 1
        self.inputs = np.zeros([size, self.inputlen, 16], dtype=float)
        self.targetlen = maxdigits+1
        self.targets = np.ones([size, self.targetlen], dtype=int)*Token.pad
        self.iscreated = [False for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not self.iscreated[idx]:
            ndigits = np.random.randint(self.mindigits, self.maxdigits+1)
            seed1 = np.random.randint(1, 10**ndigits)
            seed2 = np.random.randint(1, 10**ndigits)
            seq, target = self.rule(seed1, seed2, self.numbers)
            y = num2vec(target, ndigits, self.lendian)
            self.targets[idx][:] = Token.pad
            self.targets[idx][:len(y)] = y
            self.targets[idx][len(y)] = Token.eos
            pos = 1
            self.inputs[idx][0][Token.delim] = 1
            for i in range(self.numbers):
                vec = num2vec(seq[i], ndigits, self.lendian)
                for j,v in enumerate(vec):
                    self.inputs[idx][pos+j][v] = 1
                self.inputs[idx][pos+ndigits][Token.delim] = 1
                pos = pos + ndigits + 1
            if pos < self.inputlen:
                self.inputs[idx][pos:,Token.pad] = 1
            self.iscreated[idx] = True

        return self.inputs[idx], self.targets[idx]

dataset = NSPDataset(fib,4)
for i in range(10):
    x,y = dataset.__getitem__(i)
    print(np.argmax(x,-1))
    print(y)