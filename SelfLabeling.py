from base_self_labeling import BaseSelfLabeling
import config
from sklearn.tree import DecisionTreeClassifier
from nonconformist.base import ClassifierAdapter
from nonconformist.cp import TcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc
import numpy as np


class SelfLabeling_TCP(BaseSelfLabeling):

    def __init__(self, init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations=10, max_count=40):
        super().__init__(init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations, max_count)

    def SelectLabeled(self, labeled_data_x, labeled_data_y, unlabeled_data_x):

        # just append train data to labeled data
        labeled_x = np.concatenate((self.init_labeled_data_x, labeled_data_x)) if len(labeled_data_x) > 0 else self.init_labeled_data_x
        labeled_y = np.concatenate((self.init_labeled_data_y, labeled_data_y)) if len(labeled_data_x) > 0 else self.init_labeled_data_y
        #

        # create model to predict with confidence and credibility
        model = ClassifierAdapter(
            DecisionTreeClassifier(random_state=config.random_state, min_samples_leaf=config.min_samples_leaf))
        nc = ClassifierNc(model, MarginErrFunc())
        model_tcp = TcpClassifier(nc, smoothing=True)
        model_tcp.fit(labeled_x, labeled_y)
        s = model_tcp.predict_conf(unlabeled_data_x)
        #

        # selection method
        labeled_ind = [i for i, a in enumerate(s) if a[1] > config.confidence and a[2] > config.credibility]
        unlabeled_ind = [i for i, a in enumerate(s) if a[1] < config.confidence or a[2] < config.credibility]

        labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x = \
            np.take(unlabeled_data_x, labeled_ind, axis=0), np.take(s.T, labeled_ind), np.take(unlabeled_data_x, unlabeled_ind, axis=0)
        #

        # compare accuracy of new labeled data
        new_labeled_x = np.concatenate((labeled_x, labeled_unlabeled_x))
        new_labeled_y = np.concatenate((labeled_y, labeled_unlabeled_y))
        # old_accuracy = evaluate_classifier(labeled_x, labeled_y, self.calibration_data)
        # new_accuracy = evaluate_classifier(new_labeled_x, new_labeled_y, self.calibration_data)
        #
        # print('divide', len(new_labeled_x) / len(labeled_x), ' old: ', len(labeled_x), ' new', len(new_labeled_x))
        # print('better:', old_accuracy < new_accuracy, ' accuracy 0: ', old_accuracy, '    1: ', new_accuracy)
        # if not old_accuracy < new_accuracy:
        #     print(len(new_labeled_x) / len(labeled_x), new_accuracy > old_accuracy)

        # is_improved = True if old_accuracy < new_accuracy else False
        is_improved = True
        # is_improved = True if 1.65 > len(new_labeled_x) / len(labeled_x) > 1.4 else False
        #

        return is_improved, labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x
