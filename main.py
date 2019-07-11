from utils import *


dataset_path = config.path_to_datasets + 'diabetes'

train_data, test_data = read_data(dataset_path)

train_data['Class'] = train_data['Class'] - 1
test_data['Class'] = test_data['Class'] - 1

test_x, test_y = divide_xy(test_data)
calibration_data=[]
labeled_data, unlabeled_data, rate_p, tuple_list_pn = split_train_set(train_data)
# print(len(calibration_data), len(labeled_data), len(unlabeled_data))

# %%
# len(unlabeled_data)

# labeled_labeling = SelfLabeling_TCP()
labeled_data_x, labeled_data_y = divide_xy(labeled_data)
unlabeled_data_x, unlabeled_data_y = divide_xy(unlabeled_data)
# labeled_data_x = pd.DataFrame(labeled_data_x_nd, columns=labeled_data.columns[:-1])

print(len(labeled_data_y), type(labeled_data_y))
# labeled_data_cp = pd.DataFrame(columns=labeled_data.columns)
# print(len(labeled_data_cp.append(labeled_data_x)))
# labeled_data_cp
#
from SelfLabeling import SelfLabeling_TCP

labeling = SelfLabeling_TCP(labeled_data_x, labeled_data_y, unlabeled_data_x, max_iterations=4, max_count=100)
aflabeled_data_x, aflabeled_data_y, afunlabeled_data_x = labeling.self_label()

import numpy as np

# TODO add labeled_unlabeled_data to labeled_data ?
labeled_x = np.concatenate((labeled_data_x, aflabeled_data_x))
labeled_y = np.concatenate((labeled_data_y, aflabeled_data_y))

print("len :", len(labeled_data_x), '        \t', evaluate_classifier(labeled_data_x, labeled_data_y, test_data))
print("len :+", len(aflabeled_data_x), "=>", len(labeled_x), '\t', evaluate_classifier(labeled_x, labeled_y, test_data))

# %%
#
# from base_self_labeling import BaseSelfLabeling
# import config
# from sklearn.tree import DecisionTreeClassifier
# from nonconformist.base import ClassifierAdapter
# from nonconformist.cp import TcpClassifier
# from nonconformist.nc import ClassifierNc, MarginErrFunc
# import numpy as np
#
# model = ClassifierAdapter(
#     DecisionTreeClassifier(random_state=config.random_state, min_samples_leaf=config.min_samples_leaf))
# nc = ClassifierNc(model, MarginErrFunc())
# model_tcp = TcpClassifier(nc)
#
# print((model_tcp.smoothing))
# # print('fun : ', (model_tcp.base_icp.nc_function.score))

# %%

from SelfLabeling import SelfLabeling_TCP
