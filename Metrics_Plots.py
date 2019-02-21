import numpy as np
import matplotlib.pyplot as plt
import pandas
from Array_Loader import label_names 

def computeAccuracy(pred, actu):
    accuracy_sum = 0
    for i in range(len(actu)):
        if (pred[i]==actu[i]):
            accuracy_sum += 1

    accuracy = accuracy_sum/float(len(actu))
    return accuracy

def computeConfMat(pred, actu, norm=False):
    # Number of classes
    K = len(np.unique(actu))
    confmat = np.zeros((K, K))

    # placing increments into matching indexes
    accuracy_sum = 0
    for i in range(len(actu)):
        confmat[actu[i]][pred[i]] += 1
        if (pred[i]==actu[i]):
            accuracy_sum += 1
    #accuracy = (actu==pred).sum()/float(len(actu))
    accuracy = accuracy_sum/float(len(actu))

    if(norm):
        confmat_norm = confmat / np.sum(confmat, axis=1)
        return confmat_norm, accuracy
    return confmat, accuracy

def plotConfMat(confmat, title='Confusion Matrix', cmap=plt.cm.gray_r):
    df_confusion = pandas.DataFrame(confmat)
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()
    #plt.savefig(title+'.png')
    
def computeTruePositive(pred, actu):
    truePositiveList = []
    for i in range(len(label_names)):
        truePositive = 0
        for j in range(len(pred)):
            if pred[j] == i and actu[j] == i:
                truePositive += 1
        truePositiveList.append(truePositive)
    return truePositiveList

def computeTrueNegative(pred, actu):
    trueNegativeList = []
    for i in range(len(label_names)):
        trueNegative = 0
        for j in range(len(pred)):
            if pred[j] != i and actu[j] != i:
                trueNegative += 1
        trueNegativeList.append(trueNegative)
    return trueNegativeList

def computeFalsePositive(pred, actu):
    falsePositiveList = []
    for i in range(len(label_names)):
        falsePositive = 0
        for j in range(len(pred)):
            if pred[j] == i and actu[j] != i:
                falsePositive += 1
        falsePositiveList.append(falsePositive)
    return falsePositive

def computeFalseNegative(pred, actu):
    falsePositiveList = []
    for i in range(len(label_names)):
        falseNegative = 0
        for j in range(len(pred)):
            if pred[j] != i and actu[j] == i:
                falseNegative += 1
        falsePositiveList.append(falseNegative)
    return falsePositiveList
    

def computePrecision(pred, actu):
    truePositive = computeTruePositive(pred, actu)
    truePositive = np.array(truePositive)
    falsePositive = computeFalsePositive(pred,actu)
    falsePositive = np.array(falsePositive)
    return truePositive/ (truePositive + falsePositive)

def computeRecall(pred, actu):
    truePositive = computeTruePositive(pred, actu)
    truePositive = np.array(truePositive)
    falseNegative = computeFalseNegative(pred,actu)
    falseNegative = np.array(falseNegative)
    return truePositive/ (truePositive + falseNegative)

def computeF1(pred, actu):
    precision = computePrecision(pred, actu)
    recall = computeRecall(pred, actu)
    return (2 * precision * recall) / (precision + recall)

def computeLiftScore(pred, actu):
    #lift = (TP/(TP+FN)(TP+FP)/(TP+TN+FP+FN)
    truePositive =  computeTruePositive(pred, actu)
    truePositive = np.array(truePositive)
    trueNegative =  computeTrueNegative(pred, actu)
    trueNegative = np.array(trueNegative)
    falsePositive = computeFalsePositive(pred, actu)
    falsePositive = np.array(falsePositive)
    falseNegative = computeFalseNegative(pred, actu)
    falseNegative = np.array(falseNegative)
    numerator = truePositive / (truePositive + falsePositive)
    denominator = (truePositive+falseNegative) / (truePositive+trueNegative+falsePositive+falseNegative)
    
    return numerator / denominator

def plot_roc_curve(pred, actu, label):
    ROC = np.zeros((len(pred),2))
    actu_multi = np.array(actu)
    pred_multi = np.array(pred)
    pred = pred_multi[:]
    actu = actu_multi[:]
    pred[pred != label] = 0
    pred[pred == label] = 1
    actu[actu != label] = 0
    actu[actu == label] = 1
    
    # Compute false positive rate for current threshold.
    FPR_t = computeFalsePositiveRate(pred, actu, 1)
    ROC[idx,0] = FPR_t

    # Compute true  positive rate for current threshold.
    TPR_t = computeTruePositiveRate(pred, actu, 1)
    ROC[idx,1] = TPR_t
    
    # Plot the ROC curve.
    fig = plt.figure(figsize=(6,6))
    plt.plot(ROC[:,0], ROC[:,1], lw=2)
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.xlabel('$FPR(t)$')
    plt.ylabel('$TPR(t)$')
    plt.grid()
    
    AUC = 0.
    for i in range(100):
        AUC += (ROC[i+1,0]-ROC[i,0]) * (ROC[i+1,1]+ROC[i,1])
    AUC *= 0.5
    
    plt.title('ROC curve, AUC = %.4f'%AUC)
    plt.show()


# pred = [1,2,1,0,1,2,1,0,0,0,1,1,2,1]
# actu = [1,1,1,2,2,2,2,2,0,0,0,0,0,1]
# confmat, acc = computeConfMat(pred,actu)
# print (confmat, acc)
# plotConfMat(confmat)
#plot_roc_curve(pred,actu,label)
#computeF1(pred, actu)
