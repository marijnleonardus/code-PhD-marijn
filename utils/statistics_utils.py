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
    def propagate_standard_error(sem_array: np.ndarray):       
        """
    General purpose error propagation for the mean of independent uncertainties.
    Propagated SEM = sqrt(sum(sem^2)/n)
    
    Args:
        sem_array (np.ndarray): Array of SEM values.
                   
    Returns:
        np.ndarray: The propagated SEM.
    """
        
        n = len(sem_array)
        return np.sqrt(np.sum(np.square(sem_array)**2))/n
    