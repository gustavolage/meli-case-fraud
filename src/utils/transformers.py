import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from optbinning import OptimalBinning

class FeaturesType(BaseEstimator, TransformerMixin):
    
    def __init__(self, dtypes):
        self.dtypes = dtypes
        self.map_dtypes = {
            np.float64: dtypes["numerical"],
            np.float32: dtypes["binary"],
            str: dtypes["categorical"],
            "datetime64[ns]": dtypes["datetime"]
        }


    def fit(self, X, y=None):
        return self
    
    def __repr__(self):
        return f"FeaturesType(dtypes={self.map_dtypes})"
    
    def transform(self, X, y=None):
        for dtype in self.map_dtypes:
            for feature in self.map_dtypes[dtype]:
                if X[feature].dtype != dtype:
                    X[feature] = X[feature].astype(dtype)        
    
        return X
    
    
class NumericMissing(BaseEstimator, TransformerMixin):
    """Class responsible for handling missing values in numerical variables.
       We will impute the missing values with the value -100.0, as it does not belong to any domain.
    """
    def __init__(self, num_features):
        self.num_features = num_features

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        for feature in self.num_features:
            X[feature] = X[feature].fillna(-100.)
    
        return X
    

class BuildFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, inference=False, ratio_features: list = None):
        """
        Initializes the BuildFeatures transformer.

        Args:
            inference (bool): Flag indicating if the transformer is used for inference. Defaults to False.
            ratio_features (list): List of columns to create ratio features. Defaults to None.
        """
        self.inference = inference
        self.ratio_features = ratio_features
    
    def fit(self, X, y=None):
        return self
    
    @staticmethod
    def create_value_ratios(X, features_list):
        """
        Creates columns of ratios between 'monto' and other numeric columns in the DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with new ratio columns added.
        """
        
        for col in features_list:
            X[f"monto_div_{col}"] = np.where(X[col] != 0, X["monto"] / X[col], np.nan)
        
        return X
    
    def create_train_features(self, X):
        """
        Creates training features from the input DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with new training features added.
        """
        X['hour'] = X['fecha'].dt.hour
        X['weekday'] = X['fecha'].dt.weekday

        X['dawn_operation'] = np.where((X['hour'].between(22, 24) | X['hour'].between(0, 5)), 1., .0)

        if self.ratio_features is not None:
            self.ratio_features.extend(['hour', 'weekday'])
            X = BuildFeatures.create_value_ratios(X, self.ratio_features)

        X['N_op'] = np.where((X['o'] == 'N') & (X['p'] == 'N'), 1., .0)

        X['o'] = X['o'].map({'N': 0., 'Y': 1., 'nan': np.nan})
        X['p'] = X['p'].map({'N': 0., 'Y': 1., 'nan': np.nan})

        X["f_lower"] = np.where(X['f'] < 0.50, 1., .0)
        X["l_lower"] = np.where(X['l'] < 140.50, 1., .0)
        X["m_lower"] = np.where(X['m'] < 4.50, 1., .0)
        X["n_lower"] = np.where(X['n'] < 0.50, 1., .0)
        
        return X
    
    def create_inference_features(self, X):
        """Creates inference features from the input DataFrame.
        """
        return X
    
    def transform(self, X, y=None):
        if self.inference:
            X_transformed = self.create_inference_features(X)
        else:
            X_transformed = self.create_train_features(X)
        return X_transformed
        
    
class OptBinningEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features
        self.bin_dict_ = {}

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Must supply the target variable 'y' to the OptBinningEncoder fit")
        # Ensures that y is a 1D array
        y = np.atleast_1d(y)
        for feature in self.features:
            binning = OptimalBinning(name=feature, dtype="categorical")
            binning.fit(X[feature], y)
            self.bin_dict_[feature] = binning
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.features:
            binning = self.bin_dict_.get(col)
            if binning is not None:
                X_transformed[col] = binning.transform(X[col], metric="woe")
            else:
                raise ValueError(f"Binning not found for column: {col}")
        return X_transformed
        
    
class Selector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features]