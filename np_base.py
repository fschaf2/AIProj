import numpy as np
import os
from PIL import Image
import random

def load_images():
    images, labels = [], []
    for i in range (10):
        numpath=os.path.join('MNIST', str(i))
        for file_name in os.listdir(numpath):
            img=Image.open(os.path.join(numpath, file_name)).convert('L')
            images.append(np.array(img, dtype=np.float32).reshape(-1)/255.0)
            labels.append(i)
    return np.array(images), np.array(labels)

def get_sets(valprop):
    print("even started")
    images, labels=load_images()
    classes=np.unique(labels)
    trainimg, trainlab, valimg, vallab = [], [], [], []
    for c in classes:
        indices=np.where(labels==c)[0]
        np.random.shuffle(indices)

        split=int(len(indices)*(1-valprop))
        trainind=indices[:split]
        valind=indices[split:]
        trainimg.append(images[trainind])
        trainlab.append(labels[trainind])
        valimg.append(images[valind])
        vallab.append(labels[valind])
    trainimg=np.concatenate(trainimg)
    trainlab=np.concatenate(trainlab)
    valimg=np.concatenate(valimg)
    vallab=np.concatenate(vallab)
    train_perm=np.random.permutation(len(trainimg))
    val_perm=np.random.permutation(len(valimg))
    return (
        trainimg[train_perm],
        trainlab[train_perm],
        valimg[val_perm],
        vallab[val_perm]
    )