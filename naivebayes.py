import np_base as base
import numpy as np


def train(trainimg, trainlab):
    priors=np.zeros(10)
    probs=np.zeros((10, trainimg.shape[1]))
    for i in range(10):
        relevant_images=trainimg[trainlab==i]
        priors[i]=len(relevant_images)/trainimg.shape[1]
        probs[i]=(np.sum(relevant_images, axis=0) + 1)/(len(relevant_images) + 2) #use laplace smoothing of 1 in case of any pixels with all 0
    return probs, priors


def do_naive_bayes():
    trainimg, trainlab, valimg, vallab = base.get_sets(0.2)
    trainimg=trainimg>0.5
    valimg=valimg>0.5
    probs, priors = train(trainimg, trainlab) #probs is prob of each pixel being 1 for each number
    log_probs=np.zeros((valimg.shape[0], 10)) #log probabilities of each number, for each image
    for i in range(10):
        p=probs[i]
        log_likelihood=valimg @ np.log(p) + (1-valimg) @ np.log(1-p)
        log_probs[:, i]=np.log(priors[i]) + log_likelihood
    preds=np.argmax(log_probs, axis=1)
    accuracy=np.mean(preds==vallab) * 100
    print(f'Predictions were {accuracy}% accurate.')

do_naive_bayes()

