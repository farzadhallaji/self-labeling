from utils import *
from SelfLabeling import *

import numpy as np

import config


def test(name, method_name, iteration):
    dataset_path = config.path_to_datasets + name

    train_data, test_data = read_data(dataset_path)

    train_data['Class'] = train_data['Class'] - 1
    test_data['Class'] = test_data['Class'] - 1

    if method_name in ['icp', 'ivap1', 'ivap2']:
        calibration_data, labeled_data, unlabeled_data, rate_p, tuple_list_pn = split_train_calibration_set(train_data)
        calibration_data_x, calibration_data_y = divide_xy(calibration_data)
        labeled_data_x, labeled_data_y = divide_xy(labeled_data)
        unlabeled_data_x, unlabeled_data_y = divide_xy(unlabeled_data)
    else:
        calibration_data_x, calibration_data_y = None, None
        labeled_data, unlabeled_data, rate_p, tuple_list_pn = split_train_set(train_data)
        labeled_data_x, labeled_data_y = divide_xy(labeled_data)
        unlabeled_data_x, unlabeled_data_y = divide_xy(unlabeled_data)
    print(len(labeled_data_y), type(labeled_data_y))

    if method_name == 'tcp':
        labeling = SelfLabeling_TCP(labeled_data_x, labeled_data_y, unlabeled_data_x, max_iterations=iteration)
    elif method_name == 'icp':
        labeling = SelfLabeling_ICP(calibration_data_x, calibration_data_y, labeled_data_x, labeled_data_y,
                                    unlabeled_data_x, max_iterations=iteration)
    elif method_name == 'ccp':
        labeling = SelfLabeling_CCP(labeled_data_x, labeled_data_y, unlabeled_data_x, max_iterations=iteration)
    elif method_name == 'bcp':
        labeling = SelfLabeling_BCP(labeled_data_x, labeled_data_y, unlabeled_data_x, max_iterations=iteration)
    elif method_name == 'acp-random':
        labeling = SelfLabeling_ACP_RandomSubSampler(labeled_data_x, labeled_data_y, unlabeled_data_x,
                                                     max_iterations=iteration)
    elif method_name == 'acp-cross':
        labeling = SelfLabeling_ACP_CrossSampler(labeled_data_x, labeled_data_y, unlabeled_data_x,
                                                 max_iterations=iteration)
    elif method_name == 'acp-boot':
        labeling = SelfLabeling_ACP_BootstrapSampler(labeled_data_x, labeled_data_y, unlabeled_data_x,
                                                     max_iterations=iteration)
    elif method_name == 'ps':
        labeling = SelfLabeling_PS(labeled_data_x, labeled_data_y, unlabeled_data_x, max_iterations=iteration)
    elif method_name == 'ir':
        labeling = SelfLabeling_IR(labeled_data_x, labeled_data_y, unlabeled_data_x, max_iterations=iteration)
    elif method_name == 'ivap1':
        labeling = SelfLabeling_IVAP_v1(calibration_data_x, calibration_data_y, labeled_data_x, labeled_data_y,
                                        unlabeled_data_x, max_iterations=iteration)
    elif method_name == 'ivap2':
        labeling = SelfLabeling_IVAP_v2(calibration_data_x, calibration_data_y, labeled_data_x, labeled_data_y,
                                        unlabeled_data_x, max_iterations=iteration)

    else:
        labeling = SelfLabeling(labeled_data_x, labeled_data_y, unlabeled_data_x, max_iterations=iteration)

    after_labeled_data_x, after_labeled_data_y, after_unlabeled_data_x = labeling.self_label()

    labeled_x = np.concatenate((labeled_data_x, after_labeled_data_x))
    labeled_y = np.concatenate((labeled_data_y, after_labeled_data_y))

    print("len :", len(labeled_data_x), '        \t', evaluate_classifier(labeled_data_x, labeled_data_y, test_data))
    print("len :+", len(after_labeled_data_x), "=>", len(labeled_x), '\t', evaluate_classifier(labeled_x, labeled_y, test_data))


config.confidence = .4
config.credibility = .55
test('bupa', 'ivap1', 10)



# # %%
#
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
# labeled_data_x, labeled_data_y = divide_xy(labeled_data)
# unlabeled_data_x, unlabeled_data_y = divide_xy(unlabeled_data)
# print(len(labeled_data_y), type(labeled_data_y))
#
# # sss = VennABERS.ScoresToMultiProbs([labeled_data_x, labeled_data_y], unlabeled_data_x)
# # p0,p1 = ScoresToMultiProbs(calibrPts,testScores)
# print(sorted([labeled_data_x, labeled_data_y]))
#
#
# # %%
#
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
# labeled_data_x, labeled_data_y = divide_xy(labeled_data)
# unlabeled_data_x, unlabeled_data_y = divide_xy(unlabeled_data)
# print(len(labeled_data_y), type(labeled_data_y))
#
# labeling = SelfLabeling_TCP(labeled_data_x, labeled_data_y, unlabeled_data_x, max_iterations=10)
# aflabeled_data_x, aflabeled_data_y, afunlabeled_data_x = labeling.self_label()
#
# labeled_x = np.concatenate((labeled_data_x, aflabeled_data_x))
# labeled_y = np.concatenate((labeled_data_y, aflabeled_data_y))
#
# print("len :", len(labeled_data_x), '        \t', evaluate_classifier(labeled_data_x, labeled_data_y, test_data))
# print("len :+", len(aflabeled_data_x), "=>", len(labeled_x), '\t', evaluate_classifier(labeled_x, labeled_y, test_data))

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
