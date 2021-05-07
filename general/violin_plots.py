import matplotlib.pyplot as plt
import numpy as np
import csv

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


def read_results_csv(file_path, row_id=0):
    dice_values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dice_values.append(float(row[row_id]))

        return dice_values

# create test data
np.random.seed(19680801)

path_file_1 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
                  '3x3_grayscale_dataset/results/' \
                  'ResUnet_lr_0.001_bs_16_grayscale_16_11_2020_19_37/' \
                  'results_evaluationtest_01_ResUnet_lr_0.001_bs_16_grayscale_16_11_2020_19_37_new.csv'

path_file_2 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
              '3x3_grayscale_dataset/results/' \
              'ResUnet_lr_1e-05_bs_16_grayscale_16_11_2020_19_32/' \
              'results_evaluationtest_01_ResUnet_lr_1e-05_bs_16_grayscale_16_11_2020_19_32_new.csv'

path_file_3 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
              'lumen_data/results/' \
              'ResUnet_lr_0.001_bs_16_hsv_14_11_2020_20_06/' \
              'results_evaluation_test_02_ResUnet_lr_0.001_bs_16_hsv_14_11_2020_20_06_.csv'

path_file_4 = '/home/nearlab/Jorge/current_work/lumen_segmentation/data/' \
              'lumen_data/results/' \
              'ResUnet_lr_0.001_bs_16_rgb_06_11_2020_00_51/' \
              'results_evaluation_test_02_ResUnet_lr_0.001_bs_16_rgb_06_11_2020_00_51_.csv'

path_file_5 = '/home/nearlab/Jorge/current_work/lumen_segmentation/' \
              'data/' \
              '3x3_grayscale_dataset/results/MaskRCNN_2/' \
              'results_evaluationtest_02_MaskRCNN_2_new.csv'

path_file_6 = '/home/nearlab/Jorge/current_work/lumen_segmentation/' \
              'data/3x3_grayscale_dataset/' \
              'results/MaskRCNN_2/' \
              'results_evaluationtest_02_MaskRCNN_2_new.csv'


data_experiment_1 = sorted(read_results_csv(path_file_1, 2))
data_experiment_2 = read_results_csv(path_file_2, 2)
data_experiment_3 = read_results_csv(path_file_3, 2)
data_experiment_4 = sorted(read_results_csv(path_file_4, 2))
data_experiment_5 = sorted(read_results_csv(path_file_5, 2))
data_experiment_6 = sorted(read_results_csv(path_file_6, 2))


#data = [data_experiment_1, data_experiment_4, data_experiment_5, data_experiment_6]
#data = [sorted(np.random.normal(0, std, 100)) for std in range(1, 5)]
data = [data_experiment_1, data_experiment_2,
        data_experiment_3, 0, 0, 0]
data_2 = [0,0,0, data_experiment_4,
        data_experiment_5, data_experiment_6]
print(np.shape(data))



fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                               figsize=(9, 5), sharey=True)

ax1.set_title('Default violin plot')
ax1.set_ylabel('Observed values')
ax1.violinplot(data)
ax1.violinplot(data_2)

ax2.set_title('Customized violin plot')
parts = ax2.violinplot(
        data, showmeans=True, showmedians=True,
        showextrema=True)
"""
for pc in parts['bodies']:
    pc.set_facecolor('#D43F3A')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile(data, [25, 50, 100], axis=1)
print(quartile1, medians, quartile3)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
"""
# set style for the axes
labels = ['A', 'B', 'C', 'D', 'E', 'F']
for ax in [ax1, ax2]:
    set_axis_style(ax, labels)

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()