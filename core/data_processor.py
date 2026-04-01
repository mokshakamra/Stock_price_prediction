import math
import numpy as np
import pandas as pd

class DataLoader():
    """A class for loading and transforming data for the LSTM model"""

    def __init__(self, filename, split, cols):
        # Load CSV
        dataframe = pd.read_csv(filename)

        # 🔥 IMPORTANT FIXES
        # Convert all selected columns to numeric (remove strings, commas, etc.)
        dataframe = dataframe[cols].apply(pd.to_numeric, errors='coerce')

        # Drop rows with NaN values
        dataframe = dataframe.dropna()

        # Train-test split
        i_split = int(len(dataframe) * split)

        # Convert to pure float numpy arrays
        self.data_train = dataframe.to_numpy(dtype=float)[:i_split]
        self.data_test  = dataframe.to_numpy(dtype=float)[i_split:]

        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):
        """Create x, y test data windows"""
        data_windows = []

        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)

        if normalise:
            data_windows = self.normalise_windows(data_windows, single_window=False)

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]

        return x, y

    def get_train_data(self, seq_len, normalise):
        """Create x, y train data windows"""
        data_x = []
        data_y = []

        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)

        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        """Yield a generator of training data"""
        i = 0

        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []

            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0

                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1

            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        """Generate next data window"""
        window = self.data_train[i:i+seq_len]

        if normalise:
            window = self.normalise_windows(window, single_window=True)[0]

        x = window[:-1]
        y = window[-1, [0]]

        return x, y

    def normalise_windows(self, window_data, single_window=False):
        """Normalise window with base value = first value"""
        normalised_data = []

        window_data = [window_data] if single_window else window_data

        for window in window_data:
            normalised_window = []

            for col_i in range(window.shape[1]):
                base = window[0, col_i]

                # 🔥 Prevent division by zero
                if base == 0:
                    base = 1e-8

                normalised_col = [
                    ((float(p) / float(base)) - 1)
                    for p in window[:, col_i]
                ]

                normalised_window.append(normalised_col)

            # reshape back
            normalised_window = np.array(normalised_window).T
            normalised_data.append(normalised_window)

        return np.array(normalised_data)
