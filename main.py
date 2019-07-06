import collections

import config
from sklearn import metrics
from scipy.io.arff import loadarff
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


def evaluate_classifier(labeled_x, labeled_y, test_Data):
    _test_x, _test_y = divide_xy(test_Data)

    decision_tree = DecisionTreeClassifier(random_state=config.random_state, min_samples_leaf=config.min_samples_leaf)
    decision_tree.fit(labeled_x, labeled_y)
    y_pred = decision_tree.predict(_test_x)

    accuracy = metrics.accuracy_score(_test_y, y_pred)

    return accuracy


def divide_xy(Data):
    # assert : class = last atr 😚
    x_test = Data.values[:, 0:-1]
    y_test = (Data.values[:, -1]).astype('int')
    return x_test, y_test  # pd.DataFrame(x_test, columns=Data.columns[:-1]), pd.DataFrame(y_test, columns=Data.columns[-1])


def read_data(path):
    train_raw_data = loadarff(path + '/train.arff')
    test_raw_data = loadarff(path + '/test.arff')

    _train_data = pd.DataFrame(train_raw_data[0])
    _test_data = pd.DataFrame(test_raw_data[0])

    _train_data['Class'] = _train_data['Class'].astype(int)
    _test_data['Class'] = _test_data['Class'].astype(int)

    return _train_data, _test_data


def get_rate_p(_train_y):
    counter = collections.Counter(_train_y)
    _tuple_list_pn = counter.most_common()
    return _tuple_list_pn[0][1] / (_tuple_list_pn[0][1] + _tuple_list_pn[1][1]), _tuple_list_pn


def split_train_set(_train_data):
    labeled, unlabeled = [], []

    size_dataset = len(_train_data)
    _train_x, _train_y = divide_xy(_train_data)
    _rate_p, _tuple_list_pn = get_rate_p(_train_y)

    size_labeled_data = round(0.1 * size_dataset)

    size_labeled_p_data = round(_rate_p * size_labeled_data)
    size_labeled_n_data = size_labeled_data - size_labeled_p_data

    labeled_index = []
    unlabeled_index = []
    selected_pl = 0
    selected_nl = 0

    for i, cls in enumerate(_train_y):
        # if data point class's == 0 😚
        if cls == _tuple_list_pn[0][0]:
            if selected_pl < size_labeled_p_data:
                labeled_index.append(i)
                selected_pl += 1
            else:
                unlabeled_index.append(i)
        else:
            if selected_nl < size_labeled_n_data:
                labeled_index.append(i)
                selected_nl += 1
            else:
                unlabeled_index.append(i)

    for i in labeled_index:
        labeled.append(_train_data.values[i])

    for i in unlabeled_index:
        unlabeled.append(_train_data.values[i])

    #     print(size_dataset , size_labeled_data , size_unlabeled_data)
    #     print(rate_p , tuple_list_pn)
    #     print(size_labeled_p_data , size_labeled_n_data)
    #     print(selected_pl/(selected_pl+selected_nl),selected_pl, selected_nl)

    return pd.DataFrame(labeled, columns=_train_data.columns), \
           pd.DataFrame(unlabeled, columns=_train_data.columns), _rate_p, _tuple_list_pn


dataset_path = config.path_to_datasets + 'bupa'

train_data, test_data = read_data(dataset_path)

train_data['Class'] = train_data['Class'] - 1
test_data['Class'] = test_data['Class'] - 1

test_x, test_y = divide_xy(test_data)
labeled_data, unlabeled_data, rate_p, tuple_list_pn = split_train_set(train_data)

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

from SelfLabeling import SelfLabeling_TCP

labeling = SelfLabeling_TCP(labeled_data_x, labeled_data_y, unlabeled_data_x, max_iterations=1)
aflabeled_data_x, aflabeled_data_y, afunlabeled_data_x = labeling.self_label()


print("len : ", len(labeled_data_x), evaluate_classifier(labeled_data_x, labeled_data_y, test_data))
print("len : ", len(aflabeled_data_x), evaluate_classifier(aflabeled_data_x, aflabeled_data_y, test_data))




