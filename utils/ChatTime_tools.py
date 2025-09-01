import re

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Discretizer:
    def __init__(self, low_limit=-1, high_limit=1, n_tokens=10002):
        self.scaler = MinMaxScaler()

        self.boundaries = np.linspace(low_limit, high_limit, n_tokens - 1)
        self.centers = (self.boundaries[1:] + self.boundaries[:-1]) / 2
        self.centers = np.concatenate((self.centers[:1], self.centers, self.centers[-1:]))

    def get_centers(self):
        return self.centers

    def discretize(self, context, fit_length=None):
        fit_length = len(context) if fit_length is None else fit_length
        self.scaler.fit(context[:fit_length].reshape(-1, 1))
        scaled_context = self.scaler.transform(context.reshape(-1, 1)).reshape(-1) - 0.5

        bin_ids = np.digitize(x=scaled_context, bins=self.boundaries, right=True)
        dispersed_context = self.centers[bin_ids]

        dispersed_context[np.isnan(context)] = np.nan

        return dispersed_context

    def inverse_discretize(self, scaled_context):
        context = self.scaler.inverse_transform(scaled_context.reshape(-1, 1) + 0.5).reshape(-1)

        return context


class Serializer:
    def __init__(self, prec=4, time_sep=" ", time_flag="###", nan_flag="Nan"):
        self.prec = prec
        self.time_sep = time_sep
        self.time_flag = time_flag
        self.nan_flag = nan_flag

    def serialize(self, context):
        serialized_context = np.array([f"{self.time_flag}{i:.{self.prec}f}{self.time_flag}" for i in context])
        serialized_context[np.isnan(context)] = f"{self.time_flag}{self.nan_flag}{self.time_flag}"
        serialized_context = self.time_sep.join(serialized_context)

        return serialized_context

    def inverse_serialize(self, serialized_context):
        pattern = rf"{self.time_flag}(.*?){self.time_flag}"
        matches = re.findall(pattern, serialized_context)

        context = []
        for num in matches:
            try:
                context.append(float(num))
            except ValueError as e:
                print(e)
                context.append(np.NaN)

        context = np.array(context)

        return context
