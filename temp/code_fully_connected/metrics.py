

import numpy as np


def ap_calculator(pred, labels, top_n=None):
    """Calculate the AP for one label class.

    Args:
      pred: a numpy 1-D array of size 'num_observations'
      storing the prediction scores.
      labels: a numpy 1-D array of size 'num_observations'
      storing the ground truth labels.

    Returns:
      The total AP for the given class.
      If n is larger than the length of the ranked list,
      the average precision will be returned.

    """
    ap = 0.0
    sortidx = sorted(range(len(pred)), key=lambda k: pred[k], reverse=True)
    numpos = np.size(np.where(labels > 0))
    if numpos == 0:
        return 0

    r = len(sortidx)
    if top_n is not None:
        numpos = min(numpos, top_n)
        r = min(r, top_n)
    
    delta_recall = 1.0 / numpos
    poscount = 0.0

    for i in range(r):
        if labels[sortidx[i]] > 0:
            poscount += 1
            ap += poscount / (i + 1) * delta_recall
    
    return ap




def map_calculator(pred, labels, top_n=None):
    """Calculate the mAP.

    Args:
      pred: a numpy 2-D array of size 'num_observations x num_classes'
      storing the prediction scores.
      labels: a numpy 2-D array of size 'num_observations x num_classes'
      storing the ground truth labels.

    Returns:
      The total mAP.
      If n is larger than the length of the ranked list,
      the average precision will be returned.

    """
    aps_list = []
    num_classes = labels.shape[1]
    for l in range(num_classes):
        aps_list.append(ap_calculator(pred[:,l], labels[:,l], top_n))
    
    return np.mean(aps_list)
