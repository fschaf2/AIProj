import np_base as base
import numpy as np
import matplotlib.pyplot as plt


#Naive Bayes

#TERMINOLOGY USED: trainimg, trainlab=array of training images, array of training labels
#valimg, vallab=array of validation images, array of validation labels

#Create probability plot
def plot_probs(probs):
    fig, axes=plt.subplots(2,5, figsize=(5,5))
    axes=axes.flatten()
    for i in range(10):
        probi=probs[i]
        image=probi.reshape(28, 28)
        axes[i].imshow(image, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        axes[i].set_title(f'Digit {i}')
        axes[i].axis('off')
    fig.suptitle("Probability Maps For Each Class", fontsize=16)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.savefig("naive_maps.png", dpi=300)
    plt.close()


#train on given dataset
#returns probabilities of each pixel being 1 on each class, as well as priors (prob of each class)
def train(trainimg, trainlab):
    trainimg=trainimg>0.5 #binarize each pixel
    priors=np.zeros(10) #placeholders
    probs=np.zeros((10, trainimg.shape[1]))
    for i in range(10):
        relevant_images=trainimg[trainlab==i] #get only the images at the current number
        priors[i]=len(relevant_images)/trainimg.shape[1] #proportion that are this number
        probs[i]=(np.sum(relevant_images, axis=0) + 1)/(len(relevant_images) + 2) #get the prob of each pixel being "on" for this digit. use laplace smoothing of 1 in case of any pixels with all 0
    plot_probs(probs) #create plot
    return probs, priors

    

#test validation images and labels with given probabilities and priors. returns accuracy.
def test(probs, priors, valimg, vallab):
    valimg=valimg>0.5 #binarize
    log_probs=np.zeros((valimg.shape[0], 10)) #placeholder for log probability table
    for i in range(10):
        p=probs[i] #get probabilities for each pixel for the current digit
        log_likelihood=valimg @ np.log(p) + (1-valimg) @ np.log(1-p) #log likelihood of this image, given current digit
        log_probs[:, i]=np.log(priors[i]) + log_likelihood #representing log probabilities of this number, for each image (no need to worry about the denominator of bayes's theorem since it doesn't affect maxes)
    preds=np.argmax(log_probs, axis=1) #number with highest prob for each image
    base.plot_cm(vallab,preds,"NaiveBayes") #make confusion matrix
    accuracy=np.mean(preds==vallab) * 100
    return accuracy


#loads data, trains, and tests it
def run_default():
    trainimg, trainlab, valimg, vallab = base.get_sets(0.2)
    probs, priors = train(trainimg, trainlab) 
    return test(probs, priors, valimg, vallab)

if __name__== "__main__":
    print(f'Naive Bayes Accuracy: {run_default()}%')
