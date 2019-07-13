from utils import *
from SelfLabeling import *

import numpy as np


# dataset_path = config.path_to_datasets + 'diabetes'
dataset_path = config.path_to_datasets + 'bupa'

train_data, test_data = read_data(dataset_path)

train_data['Class'] = train_data['Class'] - 1
test_data['Class'] = test_data['Class'] - 1

test_x, test_y = divide_xy(test_data)
labeled_data, unlabeled_data, rate_p, tuple_list_pn = split_train_set(train_data)

labeled_data_x, labeled_data_y = divide_xy(labeled_data)
unlabeled_data_x, unlabeled_data_y = divide_xy(unlabeled_data)
print(len(labeled_data_y), type(labeled_data_y))

labeling = SelfLabeling_TCP(labeled_data_x, labeled_data_y, unlabeled_data_x, max_iterations=10)
aflabeled_data_x, aflabeled_data_y, afunlabeled_data_x = labeling.self_label()


labeled_x = np.concatenate((labeled_data_x, aflabeled_data_x))
labeled_y = np.concatenate((labeled_data_y, aflabeled_data_y))

print("len :", len(labeled_data_x), '        \t', evaluate_classifier(labeled_data_x, labeled_data_y, test_data))
print("len :+", len(aflabeled_data_x), "=>", len(labeled_x), '\t', evaluate_classifier(labeled_x, labeled_y, test_data))

# %%
# %%
# # dataset_path = config.path_to_datasets + 'diabetes'
# dataset_path = config.path_to_datasets + 'bupa'
#
# train_data, test_data = read_data(dataset_path)
#
# train_data['Class'] = train_data['Class'] - 1
# test_data['Class'] = test_data['Class'] - 1
#
# test_x, test_y = divide_xy(test_data)
# labeled_data, unlabeled_data, rate_p, tuple_list_pn = split_train_set(train_data)
#
# calibration_data, labeled_data, unlabeled_data, rate_p, tuple_list_pn = split_train_calibration_set(train_data)
# calibration_data_x, calibration_data_y = divide_xy(calibration_data)
# labeled_data_x, labeled_data_y = divide_xy(labeled_data)
# unlabeled_data_x, unlabeled_data_y = divide_xy(unlabeled_data)
#
# labeling = SelfLabeling_ICP(calibration_data_x, calibration_data_y,
#                             labeled_data_x, labeled_data_y, unlabeled_data_x, max_iterations=2)
# aflabeled_data_x, aflabeled_data_y, afunlabeled_data_x = labeling.self_label()
#
#
# labeled_x = np.concatenate((labeled_data_x, aflabeled_data_x))
# labeled_y = np.concatenate((labeled_data_y, aflabeled_data_y))
#
# print("len :", len(labeled_data_x), '        \t', evaluate_classifier(labeled_data_x, labeled_data_y, test_data))
# print("len :+", len(aflabeled_data_x), "=>", len(labeled_x), '\t', evaluate_classifier(labeled_x, labeled_y, test_data))
