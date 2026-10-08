"""Read bundled MNIST and the exact saved split; normalize only selected images."""
import gzip, struct
import numpy as np
import torch
from .runtime import ROOT, sha256


def read_idx(name):
    path = ROOT/'Flow_Group_Package/data/MNIST/raw'/name
    with gzip.open(path,'rb') as source: raw=source.read()
    magic=struct.unpack('>I',raw[:4])[0]
    if magic == 2051:
        count,rows,cols=struct.unpack('>III',raw[4:16])
        values=np.frombuffer(raw,dtype=np.uint8,offset=16)
        if (rows,cols)!=(28,28) or values.size!=count*rows*cols: raise ValueError('Invalid IDX images')
        return torch.from_numpy(values.reshape(count,1,rows,cols).copy())
    if magic == 2049:
        count=struct.unpack('>I',raw[4:8])[0]
        values=np.frombuffer(raw,dtype=np.uint8,offset=8)
        if values.size!=count: raise ValueError('Invalid IDX labels')
        return torch.from_numpy(values.copy()).long()
    raise ValueError('Unknown IDX magic')


def training_data():
    images=read_idx('train-images-idx3-ubyte.gz');labels=read_idx('train-labels-idx1-ubyte.gz')
    path=ROOT/'Flow_Group_Package/data/mnist_split_seed42.npz'
    with np.load(path) as split: train,valid=split['train'].copy(),split['validation'].copy()
    joined=np.concatenate([train,valid])
    if len(images)!=60000 or len(labels)!=60000 or len(train)!=55000 or len(valid)!=5000 or not np.array_equal(np.sort(joined),np.arange(60000)):
        raise ValueError('Unexpected MNIST data/split')
    return images,labels,train,valid,sha256(path)


def examples(digit=3,count=8):
    if not isinstance(digit,int) or isinstance(digit,bool) or not 0<=digit<=9 or count<1: raise ValueError('Invalid digit/count')
    images,labels,train,_,_=training_data()
    ids=torch.as_tensor(train)[labels[train]==digit][:count]
    if len(ids)!=count: raise ValueError('Too many examples requested')
    return images[ids].float()/127.5-1, labels[ids], ids
