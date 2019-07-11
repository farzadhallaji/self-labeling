from abc import ABC, abstractmethod
import numpy as np


class BaseSelfLabeling(ABC):

    def __init__(self, init_labeled_data_x, init_labeled_data_y, unlabeled_data_x, max_iterations=10, max_count=40):
        # self.calibration_data = calibration_data
        self.init_labeled_data_x = init_labeled_data_x
        self.init_labeled_data_y = init_labeled_data_y
        self.unlabeled_data_x = unlabeled_data_x
        self.max_iterations = max_iterations
        self.max_count = max_count

    @abstractmethod
    def SelectLabeled(self, labeled_data_x, labeled_data_y, unlabeled_data_x):
        """
        specific selection metric and model
        input:  train_data , labeled_data , unlabeled_data_x
        output: selected_unlabeled_data_to_label , unselected_unlabeled_data_to_label
        :return selected labeled data and remained unlabeled data
        """

    def self_label(self):
        Iteration = 0
        count = 0
        labeled_data_x, labeled_data_y = [], []
        unlabeled_data_x = self.unlabeled_data_x
        while Iteration < self.max_iterations and len(unlabeled_data_x) > 0 and count < self.max_count:

            is_improved, labeled_unlabeled_x, labeled_unlabeled_y, unlabeled_data_x = self.SelectLabeled(labeled_data_x, labeled_data_y, unlabeled_data_x)

            if is_improved:
                labeled_data_x = np.concatenate((labeled_data_x, labeled_unlabeled_x)) if len(labeled_data_x) > 0 else labeled_unlabeled_x
                labeled_data_y = np.concatenate((labeled_data_y, labeled_unlabeled_y)) if len(labeled_data_x) > 0 else labeled_unlabeled_y

                Iteration += 1
            count += 1
        return labeled_data_x, labeled_data_y, unlabeled_data_x
