import np_base as base
import numpy as np


def train(trainimg, trainlab):
    priors=np.zeros(10) #placeholders
    probs=np.zeros((10, trainimg.shape[1]))
    for i in range(10):
        relevant_images=trainimg[trainlab==i] #get only the images at the current number
        priors[i]=len(relevant_images)/trainimg.shape[1] #proportion that are this number
        probs[i]=(np.sum(relevant_images, axis=0) + 1)/(len(relevant_images) + 2) #get the prob of each pixel being "on" for this digit. use laplace smoothing of 1 in case of any pixels with all 0
    return probs, priors

    

def test(probs, priors, valimg, vallab):
    log_probs=np.zeros((valimg.shape[0], 10)) #placeholder for log probability table
    for i in range(10):
        p=probs[i] #get probabilities for each pixel for the current digit
        log_likelihood=valimg @ np.log(p) + (1-valimg) @ np.log(1-p) #log likelihood of this image, given current digit
        log_probs[:, i]=np.log(priors[i]) + log_likelihood #representing log probabilities of this number, for each image (no need to worry about the denominator of bayes's theorem since it doesn't affect maxes)
    preds=np.argmax(log_probs, axis=1) #number with highest prob for each image
    accuracy=np.mean(preds==vallab) * 100
    return accuracy


def run_default():
    trainimg, trainlab, valimg, vallab = base.get_sets(0.2)
    trainimg=trainimg>0.5 #binarize each pixel
    valimg=valimg>0.5
    probs, priors = train(trainimg, trainlab) #probs is prob of each pixel being 1 for each number, priors is prob of each number in general
    return test(probs, priors, valimg, vallab)

if __name__== "__main__":
    print(f'Naive Bayes Accuracy: {run_default()}%')

