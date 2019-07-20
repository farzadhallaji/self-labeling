from base_self_labeling import BaseSelfLabeling
import config

from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

from nonconformist.cp import TcpClassifier
from nonconformist.nc import MarginErrFunc
from nonconformist.base import ClassifierAdapter
from nonconformist.nc import ClassifierNc
from nonconformist.icp import IcpClassifier
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler, CrossSampler, RandomSubSampler
from nonconformist.acp import BootstrapConformalClassifier
from nonconformist.acp import CrossConformalClassifier

from VennPredictor.ivap import ivap

import numpy as np


class SelfLabeling(BaseSelfLabeling):

    def __init__(self, init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations=10):
        super().__init__(init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations)

    def SelectLabeled(self, labeled_data_x, labeled_data_y, unlabeled_data_x):
        # just append train data to labeled data
        labeled_x = np.concatenate((self.init_labeled_data_x, labeled_data_x)) \
            if len(labeled_data_x) > 0 else self.init_labeled_data_x
        labeled_y = np.concatenate((self.init_labeled_data_y, labeled_data_y)) \
            if len(labeled_data_x) > 0 else self.init_labeled_data_y
        #

        # create model to predict with confidence and credibility
        model = DecisionTreeClassifier(random_state=config.random_state, min_samples_leaf=config.min_samples_leaf)

        model.fit(labeled_x, labeled_y)
        s = model.predict_proba(unlabeled_data_x)
        unlabeled_data_y = model.predict(unlabeled_data_x)
        #
        # print(s)

        # selection method
        labeled_ind = [i for i, a in enumerate(s) if max(a) > config.confidence]
        unlabeled_ind = [i for i, a in enumerate(s) if max(a) < config.confidence]
        #
        labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x = \
            np.take(unlabeled_data_x, labeled_ind, axis=0), \
            np.take(unlabeled_data_y, labeled_ind), np.take(unlabeled_data_x, unlabeled_ind, axis=0)

        return labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x


class SelfLabeling_TCP(BaseSelfLabeling):

    def __init__(self, init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations=10):
        super().__init__(init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations)

    def SelectLabeled(self, labeled_data_x, labeled_data_y, unlabeled_data_x):
        # just append train data to labeled data
        labeled_x = np.concatenate((self.init_labeled_data_x, labeled_data_x)) \
            if len(labeled_data_x) > 0 else self.init_labeled_data_x
        labeled_y = np.concatenate((self.init_labeled_data_y, labeled_data_y)) \
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
        labeled_x = np.concatenate((self.init_labeled_data_x, labeled_data_x)) \
            if len(labeled_data_x) > 0 else self.init_labeled_data_x
        labeled_y = np.concatenate((self.init_labeled_data_y, labeled_data_y)) \
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
        labeled_x = np.concatenate((self.init_labeled_data_x, labeled_data_x)) \
            if len(labeled_data_x) > 0 else self.init_labeled_data_x
        labeled_y = np.concatenate((self.init_labeled_data_y, labeled_data_y)) \
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


class SelfLabeling_BCP(BaseSelfLabeling):

    def __init__(self, init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations=10):
        super().__init__(init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations)

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
        model_bcp = BootstrapConformalClassifier(IcpClassifier(ClassifierNc(model), smoothing=True))
        model_bcp.fit(labeled_x, labeled_y)
        s = model_bcp.predict(unlabeled_data_x)
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


class SelfLabeling_ACP_CrossSampler(BaseSelfLabeling):

    def __init__(self, init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations=10):
        super().__init__(init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations)

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
        model_acp = AggregatedCp(IcpClassifier(ClassifierNc(model), smoothing=True), CrossSampler())
        model_acp.fit(labeled_x, labeled_y)
        s = model_acp.predict(unlabeled_data_x)
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


class SelfLabeling_ACP_BootstrapSampler(BaseSelfLabeling):

    def __init__(self, init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations=10):
        super().__init__(init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations)

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
        model_acp = AggregatedCp(IcpClassifier(ClassifierNc(model), smoothing=True), BootstrapSampler())
        model_acp.fit(labeled_x, labeled_y)
        s = model_acp.predict(unlabeled_data_x)
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


class SelfLabeling_ACP_RandomSubSampler(BaseSelfLabeling):

    def __init__(self, init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations=10):
        super().__init__(init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations)

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
        model_acp = AggregatedCp(IcpClassifier(ClassifierNc(model), smoothing=True), RandomSubSampler())
        model_acp.fit(labeled_x, labeled_y)
        s = model_acp.predict(unlabeled_data_x)
        # print(s)
        #

        # selection method
        labeled_ind = [i for i, a in enumerate(s) if 1 - a.min() > config.confidence and a.max() > config.credibility]
        unlabeled_ind = [i for i, a in enumerate(s) if 1 - a.min() < config.confidence or a.max() < config.credibility]

        labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x = \
            np.take(unlabeled_data_x, labeled_ind, axis=0), np.take(s.argmax(axis=1), labeled_ind), np.take(
                unlabeled_data_x, unlabeled_ind, axis=0)
        #

        return labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x


class SelfLabeling_PS(BaseSelfLabeling):

    def __init__(self, init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations=10):
        super().__init__(init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations)

    def SelectLabeled(self, labeled_data_x, labeled_data_y, unlabeled_data_x):
        # just append train data to labeled data
        labeled_x = np.concatenate((self.init_labeled_data_x, labeled_data_x)) \
            if len(labeled_data_x) > 0 else self.init_labeled_data_x
        labeled_y = np.concatenate((self.init_labeled_data_y, labeled_data_y)) \
            if len(labeled_data_x) > 0 else self.init_labeled_data_y
        #

        # create model to predict with confidence and credibility
        model = DecisionTreeClassifier(random_state=config.random_state, min_samples_leaf=config.min_samples_leaf)
        model.fit(labeled_x, labeled_y)

        unlabeled_data_y = model.predict(unlabeled_data_x)

        model_ps = CalibratedClassifierCV(base_estimator=model, method='sigmoid', cv='prefit')
        s = model_ps.predict_proba(unlabeled_data_x)
        #
        # print(s)

        # selection method
        labeled_ind = [i for i, a in enumerate(s) if max(a) > config.confidence]
        unlabeled_ind = [i for i, a in enumerate(s) if max(a) < config.confidence]
        #
        labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x = \
            np.take(unlabeled_data_x, labeled_ind, axis=0), \
            np.take(unlabeled_data_y, labeled_ind), np.take(unlabeled_data_x, unlabeled_ind, axis=0)

        return labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x


class SelfLabeling_IR(BaseSelfLabeling):

    def __init__(self, init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations=10):
        super().__init__(init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations)

    def SelectLabeled(self, labeled_data_x, labeled_data_y, unlabeled_data_x):
        # just append train data to labeled data
        labeled_x = np.concatenate((self.init_labeled_data_x, labeled_data_x)) \
            if len(labeled_data_x) > 0 else self.init_labeled_data_x
        labeled_y = np.concatenate((self.init_labeled_data_y, labeled_data_y)) \
            if len(labeled_data_x) > 0 else self.init_labeled_data_y
        #

        # create model to predict with confidence and credibility
        model = DecisionTreeClassifier(random_state=config.random_state, min_samples_leaf=config.min_samples_leaf)
        model.fit(labeled_x, labeled_y)

        unlabeled_data_y = model.predict(unlabeled_data_x)

        model_ir = CalibratedClassifierCV(base_estimator=model, method='isotonic', cv='prefit')
        s = model_ir.predict_proba(unlabeled_data_x)
        #
        # print(s)

        # selection method
        labeled_ind = [i for i, a in enumerate(s) if max(a) > config.confidence]
        unlabeled_ind = [i for i, a in enumerate(s) if max(a) < config.confidence]
        #
        labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x = \
            np.take(unlabeled_data_x, labeled_ind, axis=0), \
            np.take(unlabeled_data_y, labeled_ind), np.take(unlabeled_data_x, unlabeled_ind, axis=0)

        return labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x


class SelfLabeling_IVAP_v1(BaseSelfLabeling):

    def __init__(self, calibration_data_x, calibration_data_y, init_labeled_data_x, init_labeled_data_y,
                 unlabeled_data_x, max_iterations=10):
        super().__init__(init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations)
        self.calibration_data_x = calibration_data_x
        self.calibration_data_y = calibration_data_y

    def SelectLabeled(self, labeled_data_x, labeled_data_y, unlabeled_data_x):
        # just append train data to labeled data
        labeled_x = np.concatenate((self.init_labeled_data_x, labeled_data_x)) \
            if len(labeled_data_x) > 0 else self.init_labeled_data_x
        labeled_y = np.concatenate((self.init_labeled_data_y, labeled_data_y)) \
            if len(labeled_data_x) > 0 else self.init_labeled_data_y
        #

        # create model to predict with confidence and credibility
        dt = DecisionTreeClassifier(random_state=config.random_state, min_samples_leaf=config.min_samples_leaf)
        model = ivap(dt, labeled_x, labeled_y, self.calibration_data_x, self.calibration_data_y, unlabeled_data_x)

        s = np.array(model.confidenceV1()).T
        # print((s))
        # print(s.shape)
        #

        # selection method
        # labeled_ind = [i for i, a in enumerate(s) if a > config.confidence]
        # unlabeled_ind = [i for i, a in enumerate(s) if a < config.confidence]

        labeled_ind = [i for i, a in enumerate(s) if 1 - a.min() > config.confidence and a.max() > config.credibility]
        unlabeled_ind = [i for i, a in enumerate(s) if 1 - a.min() < config.confidence or a.max() < config.credibility]

        labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x = \
            np.take(unlabeled_data_x, labeled_ind, axis=0), np.take(s.argmax(axis=1), labeled_ind)\
                , np.take(unlabeled_data_x, unlabeled_ind, axis=0)
        # print('labeled_unlabeled_y', labeled_unlabeled_y)
        return labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x


class SelfLabeling_IVAP_v2(BaseSelfLabeling):

    def __init__(self, calibration_data_x, calibration_data_y, init_labeled_data_x, init_labeled_data_y,
                 unlabeled_data_x, max_iterations=10):
        super().__init__(init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations)
        self.calibration_data_x = calibration_data_x
        self.calibration_data_y = calibration_data_y

    def SelectLabeled(self, labeled_data_x, labeled_data_y, unlabeled_data_x):
        # just append train data to labeled data
        labeled_x = np.concatenate((self.init_labeled_data_x, labeled_data_x)) \
            if len(labeled_data_x) > 0 else self.init_labeled_data_x
        labeled_y = np.concatenate((self.init_labeled_data_y, labeled_data_y)) \
            if len(labeled_data_x) > 0 else self.init_labeled_data_y
        #

        # create model to predict with confidence and credibility
        dt = DecisionTreeClassifier(random_state=config.random_state, min_samples_leaf=config.min_samples_leaf)
        model = ivap(dt, labeled_x, labeled_y, self.calibration_data_x, self.calibration_data_y, unlabeled_data_x)

        s = np.array(model.confidenceV1()).T
        # print(len(s))
        # print(s)
        #

        # selection method
        # labeled_ind = [i for i, a in enumerate(s) if a > config.confidence]
        # unlabeled_ind = [i for i, a in enumerate(s) if a < config.confidence]

        labeled_ind = [i for i, a in enumerate(s) if 1 - a.min() > config.confidence and a.max() > config.credibility]
        unlabeled_ind = [i for i, a in enumerate(s) if 1 - a.min() < config.confidence or a.max() < config.credibility]

        labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x = \
            np.take(unlabeled_data_x, labeled_ind, axis=0), np.take(s.argmax(axis=1), labeled_ind) \
                , np.take(unlabeled_data_x, unlabeled_ind, axis=0)
        # print('labeled_unlabeled_y', labeled_unlabeled_y)
        return labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x
