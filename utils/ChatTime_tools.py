import re

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Discretizer:
    """
    A discretizer class for converting continuous time series data into discrete tokens.
    
    This class handles the discretization of time series values into a fixed number of tokens,
    which is useful for language model-based time series forecasting approaches like ChatTime.
    It uses MinMaxScaler for normalization and creates discrete bins for quantization.
    
    Attributes:
        scaler (MinMaxScaler): Sklearn's MinMaxScaler for normalizing input data
        boundaries (numpy.ndarray): Array of boundary values for discretization bins
        centers (numpy.ndarray): Array of center values for each discretization bin
    """
    def __init__(self, low_limit=-1, high_limit=1, n_tokens=10002):
        """
        Initialize the Discretizer with specified parameters.
        
        Args:
            low_limit (float, optional): Lower bound for the discretization range. Defaults to -1.
            high_limit (float, optional): Upper bound for the discretization range. Defaults to 1.
            n_tokens (int, optional): Number of discrete tokens/bins to create. Defaults to 10002.
                                    This creates n_tokens-1 boundaries and n_tokens+1 centers.
        """
        self.scaler = MinMaxScaler()

        self.boundaries = np.linspace(low_limit, high_limit, n_tokens - 1)
        self.centers = (self.boundaries[1:] + self.boundaries[:-1]) / 2
        self.centers = np.concatenate((self.centers[:1], self.centers, self.centers[-1:]))

    def get_centers(self):
        """
        Get the center values for each discretization bin.
        
        Returns:
            numpy.ndarray: Array of center values for each bin, used for inverse discretization.
        """
        return self.centers

    def discretize(self, context, fit_length=None):
        """
        Discretize continuous time series data into discrete tokens.
        
        This method normalizes the input data using MinMaxScaler, then maps the values
        to discrete bins based on the predefined boundaries. The process involves:
        1. Fitting the scaler on the specified portion of the data
        2. Transforming and scaling the data to [-0.5, 0.5] range
        3. Mapping scaled values to discrete bins using digitization
        4. Replacing digitized values with bin centers
        
        Args:
            context (numpy.ndarray): Input time series data to be discretized.
                                   Should be a 1D array of continuous values.
            fit_length (int, optional): Number of data points from the beginning to use
                                      for fitting the scaler. If None, uses the entire
                                      context length. Defaults to None.
        
        Returns:
            numpy.ndarray: Discretized context where each value is replaced by the
                          center value of its corresponding bin. NaN values in the
                          original data are preserved as NaN.
        """
        fit_length = len(context) if fit_length is None else fit_length
        self.scaler.fit(context[:fit_length].reshape(-1, 1))
        scaled_context = self.scaler.transform(context.reshape(-1, 1)).reshape(-1) - 0.5

        bin_ids = np.digitize(x=scaled_context, bins=self.boundaries, right=True)
        dispersed_context = self.centers[bin_ids]

        dispersed_context[np.isnan(context)] = np.nan

        return dispersed_context

    def inverse_discretize(self, scaled_context):
        """
        Convert discretized data back to continuous values.
        
        This method reverses the discretization process by applying the inverse
        transformation of the fitted scaler to convert discrete bin centers
        back to the original data scale.
        
        Args:
            scaled_context (numpy.ndarray): Discretized context data with values
                                          corresponding to bin centers from the
                                          discretization process.
        
        Returns:
            numpy.ndarray: Continuous values in the original data scale, obtained
                          by applying inverse scaling transformation.
        """
        context = self.scaler.inverse_transform(scaled_context.reshape(-1, 1) + 0.5).reshape(-1)

        return context


class Serializer:
    """
    A serializer class for converting numerical time series data to text format and vice versa.
    
    This class handles the conversion between numerical arrays and text representations,
    which is essential for language model-based time series forecasting. It uses special
    tokens to mark numerical values and handles NaN values appropriately.
    
    Attributes:
        prec (int): Precision for floating-point number formatting
        time_sep (str): Separator string between different time points
        time_flag (str): Flag/marker to wrap around numerical values  
        nan_flag (str): Special token to represent NaN values
    """
    def __init__(self, prec=4, time_sep=" ", time_flag="###", nan_flag="Nan"):
        """
        Initialize the Serializer with formatting parameters.
        
        Args:
            prec (int, optional): Precision for floating-point formatting (number of
                                decimal places). Defaults to 4.
            time_sep (str, optional): Separator string used between serialized time points.
                                    Defaults to " " (single space).
            time_flag (str, optional): Special marker string that wraps around numerical
                                     values in the serialized format. Defaults to "###".
            nan_flag (str, optional): Special token used to represent NaN values in the
                                    serialized format. Defaults to "Nan".
        """
        self.prec = prec
        self.time_sep = time_sep
        self.time_flag = time_flag
        self.nan_flag = nan_flag

    def serialize(self, context):
        """
        Convert numerical time series data to a text string representation.
        
        This method transforms a numpy array of numerical values into a text format
        suitable for language models. Each number is formatted with the specified
        precision and wrapped with time flags. NaN values are replaced with the
        special nan_flag token.
        
        Args:
            context (numpy.ndarray): Input array of numerical time series values.
                                   Can contain NaN values which will be handled specially.
        
        Returns:
            str: Serialized string representation where each number is formatted as
                 "{time_flag}{formatted_number}{time_flag}" and separated by time_sep.
                 NaN values become "{time_flag}{nan_flag}{time_flag}".
        
        Example:
            Input: [1.2345, np.nan, 2.6789]
            Output: "###1.2345### ###Nan### ###2.6789###" (with default parameters)
        """
        serialized_context = np.array([f"{self.time_flag}{i:.{self.prec}f}{self.time_flag}" for i in context])
        serialized_context[np.isnan(context)] = f"{self.time_flag}{self.nan_flag}{self.time_flag}"
        serialized_context = self.time_sep.join(serialized_context)

        return serialized_context

    def inverse_serialize(self, serialized_context):
        """
        Convert a serialized text string back to numerical time series data.
        
        This method reverses the serialization process by extracting numerical values
        from the text format using regular expressions. It parses values wrapped in
        time flags and handles special nan_flag tokens appropriately.
        
        Args:
            serialized_context (str): Serialized string containing time series data
                                    in the format produced by the serialize method.
                                    Expected format: "{time_flag}{value}{time_flag} ..."
        
        Returns:
            numpy.ndarray: Array of numerical values extracted from the serialized string.
                          NaN values are properly restored for entries that contained
                          the nan_flag token or caused ValueError during parsing.
        
        Example:
            Input: "###1.2345### ###Nan### ###2.6789###"
            Output: [1.2345, np.nan, 2.6789]
        """
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
