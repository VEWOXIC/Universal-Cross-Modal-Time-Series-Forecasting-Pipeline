import numpy as np


def RSE(pred, true):
    """
    Calculate Root Relative Squared Error (RSE).
    
    RSE measures the relative squared error normalized by the variance of the true values.
    It provides a scale-independent measure of prediction accuracy.
    
    Args:
        pred (array-like): Predicted values from the model.
        true (array-like): True/actual values from the dataset.
    
    Returns:
        float: RSE value. Lower values indicate better predictions.
               A value of 1.0 would mean the model performs as well as
               predicting the mean of the true values.
    """
    pred, true = check(pred, true)
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    """
    Calculate the correlation coefficient between predictions and true values.
    
    This function computes the Pearson correlation coefficient, which measures
    the linear relationship between predicted and actual values.
    
    Args:
        pred (array-like): Predicted values from the model.
        true (array-like): True/actual values from the dataset.
    
    Returns:
        float: Correlation coefficient scaled by 0.01. Values range from -1 to 1,
               where 1 indicates perfect positive correlation, -1 indicates perfect
               negative correlation, and 0 indicates no linear correlation.
    """
    pred, true = check(pred, true)
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    """
    Calculate Mean Absolute Error (MAE).
    
    MAE measures the average magnitude of errors between predicted and actual values,
    treating all errors equally regardless of their direction.
    
    Args:
        pred (array-like): Predicted values from the model.
        true (array-like): True/actual values from the dataset.
    
    Returns:
        float: MAE value. Lower values indicate better predictions.
               The value is in the same units as the original data.
    """
    pred, true = check(pred, true)
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """
    Calculate Mean Squared Error (MSE).
    
    MSE measures the average of squared differences between predicted and actual values.
    It penalizes larger errors more heavily than smaller ones due to the squaring operation.
    
    Args:
        pred (array-like): Predicted values from the model.
        true (array-like): True/actual values from the dataset.
    
    Returns:
        float: MSE value. Lower values indicate better predictions.
               The value is in squared units of the original data.
    """
    pred, true = check(pred, true)
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """
    Calculate Root Mean Squared Error (RMSE).
    
    RMSE is the square root of MSE, providing error measurement in the same
    units as the original data while still penalizing larger errors more heavily.
    
    Args:
        pred (array-like): Predicted values from the model.
        true (array-like): True/actual values from the dataset.
    
    Returns:
        float: RMSE value. Lower values indicate better predictions.
               The value is in the same units as the original data.
    """
    pred, true = check(pred, true)
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    MAPE expresses accuracy as a percentage, making it easy to understand and compare
    across different datasets. It measures the average absolute percentage difference
    between predicted and actual values.
    
    Args:
        pred (array-like): Predicted values from the model.
        true (array-like): True/actual values from the dataset. Should not contain zeros
                          to avoid division by zero.
    
    Returns:
        float: MAPE value as a decimal (not percentage). Lower values indicate better
               predictions. Multiply by 100 to get percentage representation.
    
    Note:
        This metric can be problematic when true values are close to zero due to
        division by zero issues.
    """
    pred, true = check(pred, true)
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    """
    Calculate Mean Squared Percentage Error (MSPE).
    
    MSPE is similar to MAPE but squares the percentage errors, thus penalizing
    larger percentage errors more heavily than smaller ones.
    
    Args:
        pred (array-like): Predicted values from the model.
        true (array-like): True/actual values from the dataset. Should not contain zeros
                          to avoid division by zero.
    
    Returns:
        float: MSPE value as a decimal squared (not percentage). Lower values indicate
               better predictions. Take square root and multiply by 100 to get
               percentage representation.
    
    Note:
        This metric can be problematic when true values are close to zero due to
        division by zero issues.
    """
    pred, true = check(pred, true)
    return np.mean(np.square((pred - true) / true))


def SMAPE(pred, true):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    SMAPE is a symmetric version of MAPE that addresses some of the issues with
    traditional MAPE by using the average of predicted and actual values in the
    denominator. This makes it more robust when dealing with values close to zero.
    
    Args:
        pred (array-like): Predicted values from the model.
        true (array-like): True/actual values from the dataset.
    
    Returns:
        float: SMAPE value as a decimal (0-2 range). Lower values indicate better
               predictions. Multiply by 100 to get percentage representation.
               Values range from 0 (perfect) to 2 (worst case).
    """
    pred, true = check(pred, true)
    numerator = np.abs(pred - true)
    denominator = np.abs(pred) + np.abs(true)
    ratio = np.where(denominator == 0, 0, 2 * numerator / denominator)
    return np.mean(ratio)


def SMSPE(pred, true):
    """
    Calculate Symmetric Mean Squared Percentage Error (SMSPE).
    
    SMSPE is a symmetric version of MSPE that uses the average of predicted and
    actual values in the denominator and squares the error terms. This provides
    a more robust alternative to MSPE for datasets with values close to zero.
    
    Args:
        pred (array-like): Predicted values from the model.
        true (array-like): True/actual values from the dataset.
    
    Returns:
        float: SMSPE value as a decimal. Lower values indicate better predictions.
               The value represents squared symmetric percentage error.
    """
    pred, true = check(pred, true)
    numerator = pred - true
    denominator = np.abs(pred) + np.abs(true)
    ratio = np.where(denominator == 0, 0, 2 * numerator / denominator)
    return np.mean(np.square(ratio))


def check(pred, true):
    """
    Validate and convert prediction and true value inputs to numpy arrays.
    
    This utility function ensures that both predicted and true values are in the
    correct numpy array format with float32 dtype for consistent metric calculations.
    
    Args:
        pred (array-like): Predicted values from the model. Can be list, numpy array,
                          or other array-like structures.
        true (array-like): True/actual values from the dataset. Can be list, numpy array,
                          or other array-like structures.
    
    Returns:
        tuple: A tuple containing (pred, true) as numpy arrays with float32 dtype.
    """
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred, dtype=np.float32)
        true = np.array(true, dtype=np.float32)
    return pred, true


def metric(pred, true):
    """
    Calculate a comprehensive set of evaluation metrics for time series forecasting.
    
    This function computes multiple evaluation metrics commonly used in time series
    forecasting to provide a comprehensive assessment of model performance from
    different perspectives.
    
    Args:
        pred (array-like): Predicted values from the model.
        true (array-like): True/actual values from the dataset.
    
    Returns:
        tuple: A tuple containing the following metrics in order:
            - mae (float): Mean Absolute Error
            - mse (float): Mean Squared Error  
            - rmse (float): Root Mean Squared Error
            - mape (float): Mean Absolute Percentage Error
            - mspe (float): Mean Squared Percentage Error
            - smape (float): Symmetric Mean Absolute Percentage Error
            - smspe (float): Symmetric Mean Squared Percentage Error
            - rse (float): Root Relative Squared Error
            - corr (float): Correlation coefficient (scaled by 0.01)
    
    Note:
        All metrics are calculated after input validation through the check() function.
        Lower values generally indicate better performance except for correlation
        where higher values (closer to 1) indicate better performance.
    """

    pred, true = check(pred, true)

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)
    smspe = SMSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, smape, smspe, rse, corr
