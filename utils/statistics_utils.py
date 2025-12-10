import numpy as np

class Stats():

    @staticmethod
    def calculate_r_squared(y_true, y_pred):
        """
        Computes coefficient of determination R^2.
        """
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res/ss_tot)
