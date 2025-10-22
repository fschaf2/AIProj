import np_base as base
import numpy as np

def eucl_dist(factor1, factor2):
    sq_1=np.sum(factor1**2, axis=1, keepdims=True) #calculate sum of factors squared for each image. keepdims here to enable broadcasting
    sq_2=np.sum(factor2**2, axis=1)
    dist = np.sqrt(sq_1 + sq_2 - 2 * factor1 @ factor2.T) #calculate euclidian distances using (a-b)^2=a^2+b^2-2ab for more efficient matrix operations
    return dist

def do_knn(k):
    batch_size=1000
    trainimg, trainlab, valimg, vallab = base.get_sets(0.2)
    ntrain=trainlab.shape[0]
    nval=vallab.shape[0]
    preds=np.zeros(nval, dtype=int) #preallocate prediction array
    for i in range(0, nval, batch_size): #process a batch of validation images
        valend=min(i+batch_size, nval)
        batchval=valimg[i:valend]
        top_dists=np.full((valend-i, k), np.inf) #placeholder for rolling list of top (nearest) distances
        top_inds=np.zeros((valend-i,k), dtype=int) #placeholder for ndices of the nearest distances
        for j in range(0, ntrain, batch_size): #process batches of training images
            trainend=min(j+batch_size, ntrain)
            batchtrain=trainimg[j:trainend]
            dist_block=eucl_dist(batchval, batchtrain)
            for r in range(batchval.shape[0]): #for each image in the batch of validation images, compare current training batch's distances and indices with the current top ones
                top_and_cur=np.concatenate([top_dists[r], dist_block[r]])
                top_cur_inds=np.concatenate([top_inds[r], np.arange(j, trainend)])
                new_mins=np.argpartition(top_and_cur, k)[:k] #re-sort with old top + new info, and update the top arrays as necessary
                top_dists[r]=top_and_cur[new_mins]
                top_inds[r]=top_cur_inds[new_mins]

        for z in range(valend-i): #make predictions for each image in batch
            cur_neigh=top_inds[z]
            cur_neigh_lab=trainlab[cur_neigh] #which numbers are the image's nearest neighbors
            cur_dists=top_dists[z]
            weights=1/(cur_dists + 0.00001) #calculate weights from neighbors on the image

            unique_labels=np.unique(cur_neigh_lab)
            lab_weights=np.array([weights[cur_neigh_lab==ul].sum() for ul in unique_labels]) #sum up weights for each neighbor digit
            preds[i+z]=unique_labels[np.argmax(lab_weights)] #make prediction and place it in overall prediction array
    accuracy=np.mean(preds==vallab) * 100
    print(f'Predictions were {accuracy}% accurate.')

do_knn(3)