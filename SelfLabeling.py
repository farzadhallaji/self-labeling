# %% library 😚
from base_self_labeling import BaseSelfLabeling
import config
from sklearn.tree import DecisionTreeClassifier
from nonconformist.base import ClassifierAdapter
from nonconformist.cp import TcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc
import numpy as np


class SelfLabeling_TCP(BaseSelfLabeling):

    def __init__(self, train_data_x, train_data_y, unlabeled_data_x, max_iterations=10):
        super().__init__(train_data_x, train_data_y, unlabeled_data_x, max_iterations)

    def SelectLabeled(self, labeled_data_x, labeled_data_y, unlabeled_data_x):
        if len(labeled_data_x) > 0:

            labeled_x = np.concatenate((self.train_data_x, labeled_data_x))
            labeled_y = np.concatenate((self.train_data_y, labeled_data_y))
        else:
            labeled_x = self.train_data_x
            labeled_y = self.train_data_y

        model = ClassifierAdapter(
            DecisionTreeClassifier(random_state=config.random_state, min_samples_leaf=config.min_samples_leaf))
        nc = ClassifierNc(model, MarginErrFunc())
        model_tcp = TcpClassifier(nc)
        model_tcp.fit(labeled_x, labeled_y)
        s = model_tcp.predict_conf(unlabeled_data_x)
        labeled_ind = [i for i, a in enumerate(s) if a[1] > config.confidence and a[2] > config.credibility]
        unlabeled_ind = [i for i, a in enumerate(s) if a[1] < config.confidence or a[2] < config.credibility]

        return np.take(unlabeled_data_x, labeled_ind, axis=0), np.take(s.T, labeled_ind), np.take(unlabeled_data_x, unlabeled_ind, axis=0)
