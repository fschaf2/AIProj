import numpy as np
import os
from PIL import Image
import zipfile
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#Numpy Base - contains basic functions for project dealing with NumPy
#Loads sets from zip and makes confusion matrix


#TERMINOLOGY USED: trainimg, trainlab=array of training images, array of training labels
#valimg, vallab=array of validation images, array of validation labels




#Load images and labels from MNIST.zip
def load_images():
    images, labels = [], []
    with zipfile.ZipFile('MNIST.zip', 'r') as zippy: #load zip
        for file_name in zippy.namelist():
            with zippy.open(file_name) as file:
                match=re.match(r'MNIST/(\d+)/.*\.png', file_name) #if file is an image in the proper folders
                if match:
                    img=Image.open(file).convert('L')
                    images.append(np.array(img, dtype=np.float32).reshape(-1)/255.0) #add image, normalized and flattened
                    labels.append(int(match.group(1))) #add label
    return np.array(images), np.array(labels)


#load images from cache if the cache is there, otherwise load from zip and make cache
def load_images_with_caching():
    if os.path.exists("mnist.npz"):
        data=np.load("mnist.npz")
        return data['images'], data['labels']
    else:
        images, labels = load_images()
        np.savez_compressed("mnist.npz", images=images, labels=labels)
        return images, labels


#get image sets (training and validation)
#valprop-proportion to be included in validation set
def get_sets(valprop):
    images, labels=load_images_with_caching()
    classes=np.unique(labels)
    trainimg, trainlab, valimg, vallab = [], [], [], []
    for c in classes: #go through each number
        indices=np.where(labels==c)[0] #select indices of relevant images
        np.random.shuffle(indices) #shuffle indices so we select different images each time
        split=int(len(indices)*(1-valprop)) #separate the training images from validation
        trainind=indices[:split] #indices of training images
        valind=indices[split:] #indices of validation images
        trainimg.append(images[trainind]) #add all images and labels to whole list
        trainlab.append(labels[trainind])
        valimg.append(images[valind])
        vallab.append(labels[valind])
    trainimg=np.concatenate(trainimg) #turn list of arrays into array
    trainlab=np.concatenate(trainlab)
    valimg=np.concatenate(valimg)
    vallab=np.concatenate(vallab)
    train_perm=np.random.permutation(len(trainimg)) #shuffle the traning and validation sets
    val_perm=np.random.permutation(len(valimg))
    return (
        trainimg[train_perm],
        trainlab[train_perm],
        valimg[val_perm],
        vallab[val_perm]
    )


#plot confusion matrix, with vallab=validation labels, preds=predictions, model=model name for title
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
    