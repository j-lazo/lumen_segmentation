
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:18:45 2020

@author: jlazo
"""
import scipy.stats as stats
import matplotlib.pyplot as plt
import csv
from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import kruskal


def contingency_table(real_val, file_1, file_2):
    # work on this, in this actually what you would need, for every dataset
    # is the value of the pixels then you can build your table as:
    #                       Classifier2 Correct,	Classifier2 Incorrect
    # Classifier1 Correct 	Yes/Yes					Yes/No
    #Classifier1 Incorrect 	No/Yes 					No/No
    #check:
    # https://machinelearningmastery.com/mcnemars-test-for-machine-learning/
    
    return [[4, 2],	 [1, 3]]

csv_path_1 = '/home/nearlab/Jorge/current_work/' \
             'lumen_segmentation/data/lumen_data/' \
             'results/compare_3Dvs2D/' \
             'results_evaluation_test_02_ResUnet_lr_0.001_bs_8_grayscale_03_11_2020_20_08_.csv'

csv_path_2 = '/home/nearlab/Jorge/current_work/' \
             'lumen_segmentation/data/lumen_data/' \
             'results/compare_3Dvs2D/' \
             'results_evaluation_test_02_3D_ResUnet_lr_0.0001_bs_8_rgb_29_11_2020_20_15_new.csv'

csv_path_3= '/home/nearlab/Jorge/current_work/' \
            'lumen_segmentation/data/lumen_data/' \
            'results/compare_3Dvs2D/' \
            'results_evaluation_test_02_3D_ResUnet_lr_0.0001_bs_16_grayscale_16_11_2020_20_17_.csv'
csv_path_4 = '/home/nearlab/Jorge/current_work/' \
             'lumen_segmentation/data/lumen_data/' \
             'results/compare_3Dvs2D/' \
             'results_evaluation_test_02_3DMaskRCNN_2_.csv'

csv_path_5= '/home/nearlab/Jorge/current_work/' \
            'lumen_segmentation/data/lumen_data/' \
            'results/compare_3Dvs2D/' \
            'results_evaluation_test_02_ResUnet_lr_0.001_bs_8_grayscale_03_11_2020_20_08_.csv'
csv_path_6 = '/home/nearlab/Jorge/current_work/' \
             'lumen_segmentation/data/lumen_data/' \
             'results/compare_3Dvs2D/' \
             'results_evaluation_test_02_ensemble_all_data_average.csv'

def read_results_csv(file_path, row_id=0):
    dice_values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dice_values.append(float(row[row_id]))
        
        return dice_values

def read_results_csv_str(file_path, row_id=0):
    dice_values = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dice_values.append(row[row_id])
        
        return dice_values

pre_image_list = read_results_csv_str(csv_path_1, 1)

parameter_data_1 = read_results_csv(csv_path_4, 2)
parameter_data_2 = read_results_csv(csv_path_6, 2)

parameter_data_3 = read_results_csv(csv_path_3, 2)        
parameter_data_4 = read_results_csv(csv_path_4, 2)

parameter_data_5 = read_results_csv(csv_path_5, 2)        
parameter_data_6 = read_results_csv(csv_path_6, 2)


#maharashtra_ages=np.concatenate((maharashtra_ages1,maharashtra_ages2))
# Paired T-Test
result = stats.ttest_ind(a=parameter_data_1,
                         b=parameter_data_2,
                         equal_var=False)
print('T-test result')
print(result)

# compare samples
stat, p = kruskal(parameter_data_1,
                  parameter_data_2)
print('Statistics=%.9f, p=%.9f' % (stat, p))
# interpret
print('otra vez')
print(stat, p)
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')


"""plt.figure()
plt.subplot(121)
plt.hist(parameter_data_1, bins = 10)
plt.subplot(122)
plt.hist(parameter_data_2, bins = 10)
plt.show()"""