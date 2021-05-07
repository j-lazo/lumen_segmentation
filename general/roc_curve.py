from sklearn.metrics import roc_curve
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os

def load_predictions(csv_file):
    labels = []
    image_name = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            labels.append((row[0]))

            image_name.append(row[2])
    return labels, image_name


def load_labels(csv_file):
    labels = []
    image_name = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            labels.append(float(row[1]))
            image_name.append(row[2])
    return labels, image_name


def main(predicted, name_model, real):

    y_results, names = load_predictions(predicted)
    y_2test, names_test = load_labels(real)
    y_test = []
    y_pred = []
    print(len(y_results), len(names), 'predicted')
    print(len(y_2test), len(names_test), 'reals')

    for i, name in enumerate(names):
        for j, other_name in enumerate(names_test):
            if name == other_name:

                y_pred.append(float(y_results[i]))
                y_test.append(float(y_2test[j]))

    print(len(y_pred))
    print(len(y_test))
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)

    auc_model = auc(fpr_keras, tpr_keras)

    #name = real[-7:]
    plt.plot([0, 1], [0, 1], 'k--')
    label = ''.join([name_model, ' ', '(AUC = {:.3f})'])
    plt.plot(fpr_keras, tpr_keras, label=label.format(auc_model))
    #plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))


    # Zoom in view of the upper left corner.
    """plt.figure()
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    #plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')"""


if __name__ == "__main__":
    plt.figure()
    real = '/home/nearlab/Jorge/projects/GNB2020/data/' \
           'Real_values_test.csv'

    # cys
    predicted_1 = '/home/nearlab/Jorge/projects/' \
                'EMBC2021/data/results/bladder/VGG_0.csv'
    name_1 = 'cys: VGG'
    predicted_2 = '/home/nearlab/Jorge/projects/' \
                'EMBC2021/data/results/bladder/ResNet_0.csv'
    name_2 = 'cys: Inception'
    predicted_3 = '/home/nearlab/Jorge/projects/' \
                'EMBC2021/data/results/bladder/Inception_0.csv'
    name_3 = 'cys: ResNet'

    main(predicted_1, name_1, real)
    main(predicted_2, name_2, real)
    main(predicted_3, name_3, real)

    # urs

    predicted_4 = '/home/nearlab/Jorge/projects/' \
                  'EMBC2021/data/results/ureter/predictions_VGG_0.csv'
    name_4 = 'urs: VGG'
    predicted_5 = '/home/nearlab/Jorge/projects/' \
                  'EMBC2021/data/results/bladder/ResNet_0.csv'
    name_5 = 'urs: Inception'
    predicted_6 = '/home/nearlab/Jorge/projects/' \
                  'EMBC2021/data/results/bladder/Inception_0.csv'
    name_6 = 'urs: ResNet'

    main(predicted_4, name_4, real)
    main(predicted_5, name_5, real)
    main(predicted_6, name_6, real)

    # cys + urs

    predicted7 = '/home/nearlab/Jorge/projects/' \
                  'EMBC2021/data/results/bladder/VGG_0.csv'
    name_7 = 'cys + urs: VGG'
    predicted_8 = '/home/nearlab/Jorge/projects/' \
                  'EMBC2021/data/results/bladder/ResNet_0.csv'
    name_8 = 'cys + urs: Inception'
    predicted_9 = '/home/nearlab/Jorge/projects/' \
                  'EMBC2021/data/results/bladder/Inception_0.csv'
    name_9 = 'cys + urs: ResNet'

    main(predicted_3, name_3, real)
    main(predicted_4, name_4, real)
    main(predicted_5, name_5, real)

    # plt.title('ROC curve')
    plt.xlabel('False positive rate (FPR)', fontsize=18)
    plt.ylabel('True positive rate (TPR)', fontsize=18)

    plt.legend(loc='best', fontsize=13)
    plt.show()
