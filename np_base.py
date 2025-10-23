import numpy as np
import os
from PIL import Image
import random
import zipfile
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_images():
    images, labels = [], []
    with zipfile.ZipFile('MNIST.zip', 'r') as zippy:
        for file_name in zippy.namelist():
            with zippy.open(file_name) as file:
                match=re.match(r'MNIST/(\d+)/.*\.png', file_name)
                if match:
                    img=Image.open(file).convert('L')
                    images.append(np.array(img, dtype=np.float32).reshape(-1)/255.0)
                    labels.append(int(match.group(1)))
    return np.array(images), np.array(labels)

def load_images_with_caching():
    if os.path.exists("mnist.npz"):
        data=np.load("mnist.npz")
        return data['images'], data['labels']
    else:
        images, labels = load_images()
        np.savez_compressed("mnist.npz", images=images, labels=labels)
        return images, labels

def get_sets(valprop):
    images, labels=load_images_with_caching()
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

def plot_cm(vallab, preds, model):
    cm=100*confusion_matrix(vallab, preds, normalize='true')
    plt.figure(figsize=(10,8))
    ax=sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=np.arange(10),
        yticklabels=np.arange(10)
    )
    for text in ax.texts:
        text.set_text(text.get_text() + '%')
    plt.title(f'Confusion Matrix for {model}', fontsize=16)
    plt.ylabel('True Digit', fontsize=14)
    plt.xlabel('Predicted Digit', fontsize=14)
    plt.savefig(f'{model}_confmat.png', dpi=300)
    