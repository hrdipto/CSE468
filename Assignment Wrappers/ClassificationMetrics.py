import Metrics_Plots

class ClassificationMetrics:
    def __init__(trainlabels, testLabels):
        self.confmat, self.accuracy = [0,0] = get_confusion_matrix_for_heatmap(trainlabels, testLabels)

    def get_confusion_matrix_for_heatmap(trainlabels, testLabels):
        computeConfMat(testLabels, trainlabels, True)

    def calculate_accuracy():
        return self.accuracy

    def calculate_precision():
        computePrecision()

    def calculate_recall():
        computeRecall()

    def calculate_f1():
        computeF1()

    def calculate_roc_values(label):
        plot_roc_curve(label)

    def calculate_lift_score():
        computeLiftScore()
