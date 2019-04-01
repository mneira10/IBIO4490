import numpy as np


def jaccard(anot, pred):
    ax, ay, aw, ah = [int(x) for x in anot.split(' ')[1:]]
    # print(pred)
    px, py, pw, ph = pred

    maxx = max(ax, px)
    maxy = max(ay, py)

    minx = min(ax+aw, px+pw)
    miny = min(ay+ah, py+ph)

    iw = minx-maxx+1
    ih = miny-maxy+1

    inter = iw*ih
    union = aw*ah+pw*ph-inter

    return inter/union


def evalImage(annotations, predictions, threshold):
    '''
    The matlab code

    #iterate through predictios in order of confidence
      #iterate through annotations and get maximum jaccard
      #if maxjaccard>0 and maxjaccard is above threshold
        #if !claimed
          #TP
          #claim
        #else 
          #FP
          #duplicate detection
      #else
        #FP

    Since the openCv viola jones code does not return confidences, 
    we cannot order by confidence. We will do the following:

    #iterate through annotations 
      #iterate through predictios and get maximum jaccard
      #if maxjaccard>0 and maxjaccard is above threshold
        #if !claimed
          #TP
          #claim
        #else 
          #FP
          #duplicate detection
      #else
        #FP
    '''
    claimed = np.zeros(len(annotations))

    TP = 0
    FP = 0
    for i, annot in enumerate(annotations):
        # print('annotation')
        maxJaccard = 0
        correctPred = None

        # find max jaccard
        for pred in predictions:

            predJaccard = jaccard(annot, pred)

            if predJaccard > maxJaccard:
                maxJaccard = predJaccard
                correctAnnot = pred

        if maxJaccard > 0 and maxJaccard > threshold:
            if not claimed[i]:
                TP += 1
                claimed[i] = 1
        #     else:
        #         FP += 1

        # else:
        #     FP += 1
    # print(claimed,threshold)
    FN = max(len(claimed)-sum(claimed),0)
    FP = max(len(predictions)-sum(claimed),0)
    # print(TP,FP,FN)

    return TP,FP,FN

def PR(TP,FP,FN):
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    return precision,recall