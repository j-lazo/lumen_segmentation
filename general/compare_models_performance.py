import os
import re
from matplotlib import pyplot as plt
import glob


def get_information(name_model):
    types_of_data = ['npy', 'rgb', 'hsv', 'gray', 'rch', 'bch', 'gch']
    splited = name_model.split('_')
    learning_rate = splited[splited.index("lr") + 1]
    batch_size = splited[splited.index("bs") + 1]
    data_type = [data for data in types_of_data if data in splited]
    date = name_model[-16:]

    return learning_rate, batch_size, data_type, date


def compare_models(dir_to_evaluate, metric_to_evaluate='none'):
    predictions_dataset = []
    model_name = dir_to_evaluate[-8:-1]
    print(model_name)
    models = sorted(os.listdir(dir_to_evaluate))

    for model in models:
        get_information(model)

        predictions_dataset = [subfolder for subfolder in
                               os.listdir(''.join([dir_to_evaluate, model, '/predictions/']))]

    print(predictions_dataset)


def main():
    dir_to_evaluate = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/phantom_lumen/results/ResUnet/'
    metric_to_evaluate = 'none'
    # options, none, lr, bs
    compare_models(dir_to_evaluate, metric_to_evaluate)


if __name__ == '__main__':
    main()