import numpy as np


def RSE(pred, true):
    pred, true = check(pred, true)
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    pred, true = check(pred, true)
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    pred, true = check(pred, true)
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    pred, true = check(pred, true)
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    pred, true = check(pred, true)
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    pred, true = check(pred, true)
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    pred, true = check(pred, true)
    return np.mean(np.square((pred - true) / true))


def SMAPE(pred, true):
    pred, true = check(pred, true)
    numerator = np.abs(pred - true)
    denominator = np.abs(pred) + np.abs(true)
    ratio = np.where(denominator == 0, 0, 2 * numerator / denominator)
    return np.mean(ratio)


def SMSPE(pred, true):
    pred, true = check(pred, true)
    numerator = pred - true
    denominator = np.abs(pred) + np.abs(true)
    ratio = np.where(denominator == 0, 0, 2 * numerator / denominator)
    return np.mean(np.square(ratio))


def check(pred, true):
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred, dtype=np.float32)
        true = np.array(true, dtype=np.float32)
    return pred, true


def metric(pred, true):

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
