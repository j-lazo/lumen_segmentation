import os
import re
from matplotlib import pyplot as plt
import glob
import csv
import pandas as pd
import numpy as np
from matplotlib.patches import Polygon


def read_results_csv(file_path, row_id=0):
    values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            values.append(row[row_id])

        return values


def compare_boxplots(data, labels, Title):

    fig, ax1 = plt.subplots(figsize=(15, 8))
    fig.canvas.set_window_title('Boxplot Comparison')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.85, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    # ax1.set_title(Title)
    # ax1.set_xlabel('Model')
    ax1.set_ylabel(Title, fontsize=15)

    # Now fill the boxes with desired colors
    # box_colors = ['darkkhaki', 'royalblue']
    box_colors = ['darkkhaki', 'royalblue']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    averages = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        averages[i] = np.average(data[i])
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 1.1
    bottom = -0.1
    ax1.set_ylim(bottom, top)

    ax1.set_xticklabels(labels, fontsize=15, weight='bold')

    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], 0.95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='large',
                 weight=weights[k], color=box_colors[k])

    # Finally, add a basic legend

    fig.text(0.10, 0.1, '---',
             backgroundcolor=box_colors[1], color='black', weight='roman',
             size='large')

    fig.text(0.10, 0.045, '--',
             backgroundcolor=box_colors[1],
             color='white', weight='roman', size='large')


    fig.text(0.10, 0.005, '*', color='white', backgroundcolor='silver',
             weight='roman', size='large')

    fig.text(0.115, 0.003, 'Average Value', color='black', weight='roman',
             size='large')

    plt.show()


def get_information(name_model, model):
    types_of_data = ['npy', 'rgb', 'hsv', 'gray', 'rch', 'bch', 'gch']
    splited = name_model.split('_')
    learning_rate = splited[splited.index("lr") + 1]
    batch_size = splited[splited.index("bs") + 1]
    data_type = [data for data in types_of_data if data in splited]
    date = name_model[-16:]
    model_information = {"model name": model,
                         "learning rate": learning_rate,
                         "batch size": batch_size,
                         "data type": data_type,
                         "training date": date,
                         "folder name": name_model}

    return model_information


def simple_comparison(list_models, title='', metric_to_evaluate='none', hyper_param_to_compare='none'):
    fig1, axs1 = plt.subplots(2, len(list_models), figsize=(17, 8))
    fig1.suptitle(title, fontsize=16)
    for j, model in enumerate(list_models):
        for i in range(model['num folders evaluation']):

            axs1[i, j].boxplot(model[metric_to_evaluate + str(i)])
            axs1[i, j].set_ylim([0, 1.1])
            axs1[i, j].set_title(''.join(['bs: ', model['batch size'], '\n', 'lr: ', "{:.1e}".format(float(model['learning rate']))]))
            if j != 0:
                axs1[i, j].set_yticks([])
            axs1[i, j].set_xticks([])


def compare_performance_metrics(list_models, hyper_param_to_compare='none'):
    print(list_models[0].keys())
    simple_comparison(list_models, title='$DSC$ comparison', metric_to_evaluate='dsc evaluation ')
    simple_comparison(list_models, title='$Prec$ comparison', metric_to_evaluate='sens evaluation ')
    simple_comparison(list_models, title='$Rec$ comparison', metric_to_evaluate='spec evaluation ')

    return 0


def compare_history(dir_to_evaluate, list_models):
    for model_information in list_models:
        model = model_information["folder name"]
        learning_rate = model_information["learning rate"]
        batch_size = model_information["batch size"]
        directory_history = [s for s in os.listdir(dir_to_evaluate + model) if 'performance_metrics_' in s][0]
        directory_history_dir = ''.join([dir_to_evaluate, model, '/', directory_history])

        train_loss_history = read_results_csv(directory_history_dir, row_id=1)[1:]
        train_loss_history = [float(loss) for loss in train_loss_history]
        train_accuracy_history = read_results_csv(directory_history_dir, row_id=2)[1:]
        train_accuracy_history = [float(train) for train in train_accuracy_history]
        train_dsc_history = read_results_csv(directory_history_dir, row_id=5)[1:]
        train_dsc_history = [float(dsc) for dsc in train_dsc_history]

        val_loss_history = read_results_csv(directory_history_dir, row_id=7)[1:]
        val_loss_history = [float(loss) for loss in val_loss_history]
        val_accuracy_history = read_results_csv(directory_history_dir, row_id=8)[1:]
        val_accuracy_history = [float(train) for train in val_accuracy_history]
        val_dsc_history = read_results_csv(directory_history_dir, row_id=11)[1:]
        val_dsc_history = [float(dsc) for dsc in val_dsc_history]

        plt.subplot(3, 2, 1)
        plt.plot(train_accuracy_history, label=''.join(['train ', 'lr:', learning_rate, ' bs: ', batch_size]))
        plt.subplot(3, 2, 3)
        plt.plot(train_dsc_history, label=''.join(['train ', 'lr:', learning_rate, ' bs: ', batch_size]))
        plt.subplot(3, 2, 5)
        plt.plot(train_loss_history, label=''.join(['train ', 'lr:', learning_rate, ' bs: ', batch_size]))

        plt.subplot(3, 2, 2)
        plt.plot(val_accuracy_history, label=''.join(['val ', 'lr:', learning_rate, ' bs: ', batch_size]))
        plt.subplot(3, 2, 4)
        plt.plot(val_dsc_history, label=''.join(['val ', 'lr:', learning_rate, ' bs: ', batch_size]))
        plt.subplot(3, 2, 6)
        plt.plot(val_loss_history, label=''.join(['val ', 'lr:', learning_rate, ' bs: ', batch_size]))

    plt.subplot(3, 2, 1).set_title('Accuracy Training')
    plt.xticks([])
    plt.subplot(3, 2, 3).set_title('DSC Training')
    plt.xticks([])
    plt.subplot(3, 2, 5).set_title('Loss Training')

    plt.subplot(3, 2, 2).set_title('Accuracy Val')
    plt.xticks([])
    plt.subplot(3, 2, 4).set_title('DSC Val')
    plt.xticks([])
    plt.subplot(3, 2, 6).set_title('Loss Val')

    plt.legend(loc='best')


def update_information(dir_to_evaluate, model_information, predictions_dataset):

    model = model_information["folder name"]
    directory_history = [s for s in os.listdir(dir_to_evaluate + model) if 'performance_metrics_' in s][0]
    results_evaluation = [s for s in os.listdir(dir_to_evaluate + model) if 'results_evaluation' in s]

    directory_history_dir = ''.join([dir_to_evaluate, model, '/', directory_history])
    train_loss_history = read_results_csv(directory_history_dir, row_id=1)[1:]
    train_loss_history = [float(loss) for loss in train_loss_history]
    train_accuracy_history = read_results_csv(directory_history_dir, row_id=2)[1:]
    train_accuracy_history = [float(train) for train in train_accuracy_history]
    train_dsc_history = read_results_csv(directory_history_dir, row_id=5)[1:]
    train_dsc_history = [float(dsc) for dsc in train_dsc_history]
    model_information['train acc history'] = train_accuracy_history
    model_information['train dsc history'] = train_dsc_history
    model_information['train loss history'] = train_loss_history
    for i, evaluation in enumerate(results_evaluation):
        evaluation_file = ''.join([dir_to_evaluate, model, '/', evaluation])
        names_files = read_results_csv(evaluation_file, 1)
        dsc_values = read_results_csv(evaluation_file, 2)
        sensitivity_values = read_results_csv(evaluation_file, 3)
        specificity_values = read_results_csv(evaluation_file, 4)

        model_information['num folders evaluation'] = len(results_evaluation)
        model_information['evaluation ' + str(i)] = evaluation
        model_information['names evaluation files ' + str(i)] = names_files
        dsc_values = [float(element) for element in dsc_values]
        model_information['dsc evaluation ' + str(i)] = dsc_values
        sensitivity_values = [float(element) for element in sensitivity_values]
        model_information['sens evaluation ' + str(i)] = sensitivity_values
        specificity_values = [float(element) for element in specificity_values]
        model_information['spec evaluation ' + str(i)] = specificity_values

    return model_information


def compare_models(dir_to_evaluate, metric_to_evaluate='none', history=True, perform_metrics=True):
    model_name = dir_to_evaluate[-8:-1]
    print(model_name)
    models = sorted(os.listdir(dir_to_evaluate))
    predictions_dataset = []
    list_models = []


    for k, model in enumerate(models):
        model_information = get_information(model, model_name)
        for folder in os.listdir(''.join([dir_to_evaluate, model, '/predictions/'])):
            if (folder not in predictions_dataset):
                predictions_dataset.append(folder)

        update_information(dir_to_evaluate, model_information, predictions_dataset)
        list_models.append(model_information)
    #if history is True:
    #    compare_history(list_models)

    if perform_metrics is True:
        compare_performance_metrics(list_models, metric_to_evaluate)

    if perform_metrics is True:
        plt.figure()
        compare_history(dir_to_evaluate, list_models)

    plt.show()


def main():
    dir_to_evaluate = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/phantom_lumen/' \
                      'results/continuous_blocks_ResUnet/'
    metric_to_evaluate = 'none'
    # options, none, lr, bs
    compare_models(dir_to_evaluate, metric_to_evaluate)


if __name__ == '__main__':
    main()