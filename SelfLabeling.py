from base_self_labeling import BaseSelfLabeling
import config
from sklearn.tree import DecisionTreeClassifier
from nonconformist.base import ClassifierAdapter
from nonconformist.cp import TcpClassifier, IcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc
from nonconformist.base import ClassifierAdapter
from nonconformist.nc import ClassifierNc
from nonconformist.icp import IcpClassifier
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler, CrossSampler, RandomSubSampler
from nonconformist.acp import BootstrapConformalClassifier
from nonconformist.acp import CrossConformalClassifier
from nonconformist.evaluation import class_mean_errors

import numpy as np


class SelfLabeling_TCP(BaseSelfLabeling):

    def __init__(self, init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations=10):
        super().__init__(init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations)

    def SelectLabeled(self, labeled_data_x, labeled_data_y, unlabeled_data_x):
        # just append train data to labeled data
        labeled_x = np.concatenate((self.init_labeled_data_x, labeled_data_x))\
            if len(labeled_data_x) > 0 else self.init_labeled_data_x
        labeled_y = np.concatenate((self.init_labeled_data_y, labeled_data_y))\
            if len(labeled_data_x) > 0 else self.init_labeled_data_y
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
            np.take(unlabeled_data_x, labeled_ind, axis=0), \
            np.take(s.T, labeled_ind), np.take(unlabeled_data_x, unlabeled_ind, axis=0)

        return labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x


class SelfLabeling_ICP(BaseSelfLabeling):

    def __init__(self, calibration_data_x, calibration_data_y, init_labeled_data_x, init_labeled_data_y,
                 unlabeled_data_x, max_iterations=10):
        super().__init__(init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations)
        self.calibration_data_x = calibration_data_x
        self.calibration_data_y = calibration_data_y

    def SelectLabeled(self, labeled_data_x, labeled_data_y, unlabeled_data_x):
        # just append train data to labeled data
        labeled_x = np.concatenate((self.init_labeled_data_x, labeled_data_x))\
            if len(labeled_data_x) > 0 else self.init_labeled_data_x
        labeled_y = np.concatenate((self.init_labeled_data_y, labeled_data_y))\
            if len(labeled_data_x) > 0 else self.init_labeled_data_y
        #

        # create model to predict with confidence and credibility
        model = ClassifierAdapter(
            DecisionTreeClassifier(random_state=config.random_state, min_samples_leaf=config.min_samples_leaf))
        nc = ClassifierNc(model, MarginErrFunc())
        model_icp = IcpClassifier(nc, smoothing=True)
        model_icp.fit(labeled_x, labeled_y)
        model_icp.calibrate(self.calibration_data_x, self.calibration_data_y)
        s = model_icp.predict_conf(unlabeled_data_x)
        print(s)
        #

        # selection method
        labeled_ind = [i for i, a in enumerate(s) if a[1] > config.confidence and a[2] > config.credibility]
        unlabeled_ind = [i for i, a in enumerate(s) if a[1] < config.confidence or a[2] < config.credibility]

        labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x = \
            np.take(unlabeled_data_x, labeled_ind, axis=0), np.take(s.T, labeled_ind), np.take(unlabeled_data_x,
                                                                                               unlabeled_ind, axis=0)
        #

        return labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x


class SelfLabeling_CCP(BaseSelfLabeling):

    def __init__(self, init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations=10):
        super().__init__(init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations)

    def SelectLabeled(self, labeled_data_x, labeled_data_y, unlabeled_data_x):
        # just append train data to labeled data
        labeled_x = np.concatenate((self.init_labeled_data_x, labeled_data_x))\
            if len(labeled_data_x) > 0 else self.init_labeled_data_x
        labeled_y = np.concatenate((self.init_labeled_data_y, labeled_data_y))\
            if len(labeled_data_x) > 0 else self.init_labeled_data_y
        #

        # create model to predict with confidence and credibility
        model = ClassifierAdapter(
            DecisionTreeClassifier(random_state=config.random_state, min_samples_leaf=config.min_samples_leaf))
        nc = ClassifierNc(model, MarginErrFunc())
        model_icp = IcpClassifier(nc, smoothing=True)
        model_ccp = CrossConformalClassifier(model_icp)
        model_ccp.fit(labeled_x, labeled_y)

        s = model_ccp.predict(unlabeled_data_x)
        # print(s)
        #

        # selection method
        labeled_ind = [i for i, a in enumerate(s) if a.max() > config.confidence and 1 - a.min() > config.credibility]
        unlabeled_ind = [i for i, a in enumerate(s) if a.max() < config.confidence or 1 - a.min() < config.credibility]

        labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x = \
            np.take(unlabeled_data_x, labeled_ind, axis=0), np.take(s.argmax(axis=1), labeled_ind), np.take(
                unlabeled_data_x, unlabeled_ind, axis=0)
        #

        return labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x


class SelfLabeling_ICP(BaseSelfLabeling):

    def __init__(self, calibration_data_x, calibration_data_y, init_labeled_data_x, init_labeled_data_y,
                 unlabeled_data_x, max_iterations=10):
        super().__init__(init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations)
        self.calibration_data_x = calibration_data_x
        self.calibration_data_y = calibration_data_y

    def SelectLabeled(self, labeled_data_x, labeled_data_y, unlabeled_data_x):
        # just append train data to labeled data
        labeled_x = np.concatenate((self.init_labeled_data_x, labeled_data_x)) if len(
            labeled_data_x) > 0 else self.init_labeled_data_x
        labeled_y = np.concatenate((self.init_labeled_data_y, labeled_data_y)) if len(
            labeled_data_x) > 0 else self.init_labeled_data_y
        #

        # create model to predict with confidence and credibility
        model = ClassifierAdapter(
            DecisionTreeClassifier(random_state=config.random_state, min_samples_leaf=config.min_samples_leaf))
        nc = ClassifierNc(model, MarginErrFunc())
        model_icp = IcpClassifier(nc, smoothing=True)
        model_icp.fit(labeled_x, labeled_y)
        model_icp.calibrate(self.calibration_data_x, self.calibration_data_y)
        s = model_icp.predict_conf(unlabeled_data_x)
        # print(s)
        #

        # selection method
        labeled_ind = [i for i, a in enumerate(s) if a[1] > config.confidence and a[2] > config.credibility]
        unlabeled_ind = [i for i, a in enumerate(s) if a[1] < config.confidence or a[2] < config.credibility]

        labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x = \
            np.take(unlabeled_data_x, labeled_ind, axis=0), np.take(s.T, labeled_ind), np.take(unlabeled_data_x,
                                                                                               unlabeled_ind, axis=0)
        #

        return labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x

    class SelfLabeling_ICP(BaseSelfLabeling):

        def __init__(self, calibration_data_x, calibration_data_y, init_labeled_data_x, init_labeled_data_y,
                     unlabeled_data_x, max_iterations=10):
            super().__init__(init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations)
            self.calibration_data_x = calibration_data_x
            self.calibration_data_y = calibration_data_y

        def SelectLabeled(self, labeled_data_x, labeled_data_y, unlabeled_data_x):
            # just append train data to labeled data
            labeled_x = np.concatenate((self.init_labeled_data_x, labeled_data_x)) if len(
                labeled_data_x) > 0 else self.init_labeled_data_x
            labeled_y = np.concatenate((self.init_labeled_data_y, labeled_data_y)) if len(
                labeled_data_x) > 0 else self.init_labeled_data_y
            #

            # create model to predict with confidence and credibility
            model = ClassifierAdapter(
                DecisionTreeClassifier(random_state=config.random_state, min_samples_leaf=config.min_samples_leaf))
            nc = ClassifierNc(model, MarginErrFunc())
            model_icp = IcpClassifier(nc, smoothing=True)
            model_icp.fit(labeled_x, labeled_y)
            model_icp.calibrate(self.calibration_data_x, self.calibration_data_y)
            s = model_icp.predict_conf(unlabeled_data_x)
            print(s)
            #

            # selection method
            labeled_ind = [i for i, a in enumerate(s) if a[1] > config.confidence and a[2] > config.credibility]
            unlabeled_ind = [i for i, a in enumerate(s) if a[1] < config.confidence or a[2] < config.credibility]

            labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x = \
                np.take(unlabeled_data_x, labeled_ind, axis=0), np.take(s.T, labeled_ind), np.take(unlabeled_data_x,
                                                                                                   unlabeled_ind,
                                                                                                   axis=0)
            #

            return labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x

        class SelfLabeling_ICP(BaseSelfLabeling):

            def __init__(self, calibration_data_x, calibration_data_y, init_labeled_data_x, init_labeled_data_y,
                         unlabeled_data_x, max_iterations=10):
                super().__init__(init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations)
                self.calibration_data_x = calibration_data_x
                self.calibration_data_y = calibration_data_y

            def SelectLabeled(self, labeled_data_x, labeled_data_y, unlabeled_data_x):
                # just append train data to labeled data
                labeled_x = np.concatenate((self.init_labeled_data_x, labeled_data_x)) if len(
                    labeled_data_x) > 0 else self.init_labeled_data_x
                labeled_y = np.concatenate((self.init_labeled_data_y, labeled_data_y)) if len(
                    labeled_data_x) > 0 else self.init_labeled_data_y
                #

                # create model to predict with confidence and credibility
                model = ClassifierAdapter(
                    DecisionTreeClassifier(random_state=config.random_state, min_samples_leaf=config.min_samples_leaf))
                nc = ClassifierNc(model, MarginErrFunc())
                model_icp = IcpClassifier(nc, smoothing=True)
                model_icp.fit(labeled_x, labeled_y)
                model_icp.calibrate(self.calibration_data_x, self.calibration_data_y)
                s = model_icp.predict_conf(unlabeled_data_x)
                print(s)
                #

                # selection method
                labeled_ind = [i for i, a in enumerate(s) if a[1] > config.confidence and a[2] > config.credibility]
                unlabeled_ind = [i for i, a in enumerate(s) if a[1] < config.confidence or a[2] < config.credibility]

                labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x = \
                    np.take(unlabeled_data_x, labeled_ind, axis=0), np.take(s.T, labeled_ind), np.take(unlabeled_data_x,
                                                                                                       unlabeled_ind,
                                                                                                       axis=0)
                #

                return labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x
