import np_base as base
import numpy as np

def eucl_dist(factor1, factor2):
    sq_1=np.sum(factor1**2, axis=1, keepdims=True)
    sq_2=np.sum(factor2**2, axis=1)
    dist = np.sqrt(sq_1 + sq_2 - 2 * factor1 @ factor2.T)
    return dist

def do_knn(k):
    batch_size=1000
    trainimg, trainlab, valimg, vallab = base.get_sets(0.2)
    ntrain=trainlab.shape[0]
    nval=vallab.shape[0]
    preds=np.zeros(nval, dtype=int)
    for i in range(0, nval, batch_size):
        valend=min(i+batch_size, nval)
        batchval=valimg[i:valend]
        top_dists=np.full((valend-i, k), np.inf)
        top_inds=np.zeros((valend-i,k), dtype=int)
        for j in range(0, ntrain, batch_size):
            trainend=min(j+batch_size, ntrain)
            batchtrain=trainimg[j:trainend]
            dist_block=eucl_dist(batchval, batchtrain)
            for r in range(batchval.shape[0]):
                top_and_cur=np.concatenate([top_dists[r], dist_block[r]])
                top_cur_inds=np.concatenate([top_inds[r], np.arange(j, trainend)])
                new_mins=np.argpartition(top_and_cur, k)[:k]
                top_dists[r]=top_and_cur[new_mins]
                top_inds[r]=top_cur_inds[new_mins]

        for z in range(valend-i):
            cur_neigh=top_inds[z]
            cur_neigh_lab=trainlab[cur_neigh]
            cur_dists=top_dists[z]
            weights=1/(cur_dists + 0.00001)

            unique_labels=np.unique(cur_neigh_lab)
            lab_weights=np.array([weights[cur_neigh_lab==ul].sum() for ul in unique_labels])
            preds[i+z]=unique_labels[np.argmax(lab_weights)]
    accuracy=np.mean(preds==vallab) * 100
    print(f'Predictions were {accuracy}% accurate.')

do_knn(3)