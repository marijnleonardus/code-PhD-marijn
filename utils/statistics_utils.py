import numpy as np


class Stats():
    @staticmethod
    def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Computes coefficient of determination R^2.
        """
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res/ss_tot)

    @staticmethod
    def weighted_sum_of_squares(y_true, y_pred, y_err):
        """
        Computes the weighted sum of squares (often called Chi-squared or chi^2).

        The weights are defined as the reciprocal of the variance (1 / y_err^2).
        This function assumes y_err represents the standard deviation (sigma).

        Args:
            y_true (np.ndarray): The true target values.
            y_pred (np.ndarray): The predicted values.
            y_err (np.ndarray): The standard deviation (error) of the true values.

        Returns:
            float: The computed weighted sum of squares (chi^2 value).

        Raises:
            ValueError: If the input arrays are not of the same length, 
                        or if any y_err is zero or less.
        """
        if not (len(y_true) == len(y_pred) == len(y_err)):
            raise ValueError("Input arrays must have the same length.")

        if np.any(y_err <= 0):
            # We need to ensure y_err is positive because we divide by its square.
            raise ValueError("All elements in y_err must be greater than zero.")
            
        residuals_squared = (y_true - y_pred)**2
        weights = 1.0/(y_err**2)
        weighted_sq_residuals = weights*residuals_squared
        chi_squared = np.sum(weighted_sq_residuals)
        return chi_squared
    
    @staticmethod
    def weighted_average_and_se(means, sems):
        """
        Calculates the inverse-variance weighted mean and propagated SE.
        Args:
            means: List of arrays (the y-values of the datasets)
            sems:  List of arrays (the SE values of the datasets
        """
        # Convert lists to 2D numpy arrays for vectorization
        means = np.array(means)
        sems = np.array(sems)
        
        # Calculate weights (w = 1 / SE^2)
        # Adding a tiny epsilon to avoid division by zero if SE is 0
        weights = 1.0/(sems**2 + 1e-15)
        
        # Calculate weighted mean
        weighted_mean = np.sum(means*weights, axis=0)/np.sum(weights, axis=0)
        
        # Calculate propagated standard error (SE = sqrt(1 / sum(weights)))
        propagated_se = np.sqrt(1.0/np.sum(weights, axis=0))
        
        return weighted_mean, propagated_se
    