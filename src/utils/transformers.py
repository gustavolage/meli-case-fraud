import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from optbinning import OptimalBinning
from typing import List, Dict, Optional, Any

#------------------
# FeaturesType
#------------------
class FeaturesType(BaseEstimator, TransformerMixin):
    """Class responsible for transforming the data types of the features in the dataset."""
    def __init__(self, dtypes: Dict[str, List[str]]) -> None:
        self.dtypes: Dict[str, List[str]] = dtypes
        self.map_dtypes: Dict[Any, List[str]] = {
            np.float64: dtypes["numerical"],
            np.float32: dtypes["binary"],
            str: dtypes["categorical"],
            "datetime64[ns]": dtypes["datetime"]
        }

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeaturesType":
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        for dtype in self.map_dtypes:
            for feature in self.map_dtypes[dtype]:
                if X[feature].dtype != dtype:
                    X[feature] = X[feature].astype(dtype)
        return X

    def __repr__(self) -> str:
        return f"FeaturesType(dtypes={self.map_dtypes})"

#------------------
# NumericMissing
#------------------
class NumericMissing(BaseEstimator, TransformerMixin):
    """Class responsible for handling missing values in numerical variables.
       We will impute the missing values with the value -100.0, as it does not belong to any domain.
    """
    def __init__(self, num_features: List[str]) -> None:
        self.num_features: List[str] = num_features

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "NumericMissing":
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        for feature in self.num_features:
            X[feature] = X[feature].fillna(-100.0)
        return X

    def __repr__(self) -> str:
        return f"NumericMissing(num_features={self.num_features})"

#------------------
# BuildFeatures
#------------------
class BuildFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, inference: bool = False, ratio_features: Optional[List[str]] = None) -> None:
        """Creates new features from the input DataFrame."""
        self.inference: bool = inference
        self.ratio_features: Optional[List[str]] = ratio_features

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BuildFeatures":
        return self

    @staticmethod
    def create_value_ratios(X: pd.DataFrame, features_list: List[str]) -> pd.DataFrame:
        """Creates columns of ratios between 'monto' and other numeric columns in the DataFrame."""
        for col in features_list:
            X[f"monto_div_{col}"] = np.where(X[col] != 0, X["monto"] / X[col], np.nan)
        return X

    def create_train_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Creates training features from the input DataFrame."""
        X['hour'] = X['fecha'].dt.hour
        X['weekday'] = X['fecha'].dt.weekday

        X['dawn_operation'] = np.where((X['hour'].between(22, 24) | X['hour'].between(0, 5)), 1.0, 0.0)

        if self.ratio_features is not None:
            # Create new list to avoid modifying the original
            features = self.ratio_features + ['hour', 'weekday']
            X = BuildFeatures.create_value_ratios(X, features)
            
        # X['N_op'] = np.where((X['o'] == 'N') & (X['p'] == 'N'), 1.0, 0.0)
            
        X["f_lower"] = np.where(X['f'] < 0.50, 1.0, 0.0)
        X["l_lower"] = np.where(X['l'] < 140.50, 1.0, 0.0)
        X["m_lower"] = np.where(X['m'] < 4.50, 1.0, 0.0)
        X["n_lower"] = np.where(X['n'] < 0.50, 1.0, 0.0)

        return X

    def create_inference_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Creates inference features from the input DataFrame."""
        X['hour'] = X['fecha'].dt.hour
        X['weekday'] = X['fecha'].dt.weekday

        X['dawn_operation'] = np.where((X['hour'].between(22, 24) | X['hour'].between(0, 5)), 1.0, 0.0)

        if self.ratio_features is not None:
            X = BuildFeatures.create_value_ratios(X, self.ratio_features)

        X['o'] = X['o'].map({'N': 0.0, 'Y': 1.0, 'nan': np.nan})
        X['p'] = X['p'].map({'N': 0.0, 'Y': 1.0, 'nan': np.nan})

        X["f_lower"] = np.where(X['f'] < 0.50, 1.0, 0.0)
        X["l_lower"] = np.where(X['l'] < 140.50, 1.0, 0.0)
        X["m_lower"] = np.where(X['m'] < 4.50, 1.0, 0.0)
        X["n_lower"] = np.where(X['n'] < 0.50, 1.0, 0.0)

        return X

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        if self.inference:
            X_transformed = self.create_inference_features(X)
        else:
            X_transformed = self.create_train_features(X)
        return X_transformed

    def __repr__(self) -> str:
        return f"BuildFeatures(inference={self.inference}, ratio_features={self.ratio_features})"

#------------------
# OptBinningEncoder
#------------------
class OptBinningEncoder(BaseEstimator, TransformerMixin):
    """Transforms the categorical features using the WoE transformation."""
    def __init__(self, features: Optional[List[str]] = None) -> None:
        self.features: Optional[List[str]] = features
        self.bin_dict_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OptBinningEncoder":
        if y is None:
            raise ValueError("Must supply the target variable 'y' to the OptBinningEncoder fit")
        # Ensures that y is a 1D array
        y = np.atleast_1d(y)
        if self.features is None:
            raise ValueError("No features provided for binning")
        for feature in self.features:
            binning = OptimalBinning(name=feature, dtype="categorical")
            binning.fit(X[feature], y)
            self.bin_dict_[feature] = binning
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()
        if self.features is None:
            raise ValueError("No features provided for transformation")
        for col in self.features:
            binning = self.bin_dict_.get(col)
            if binning is not None:
                X_transformed[col] = binning.transform(X[col], metric="woe")
            else:
                raise ValueError(f"Binning not found for column: {col}")
        return X_transformed

    def __repr__(self) -> str:
        return f"OptBinningEncoder(features={self.features}, bin_dict_keys={list(self.bin_dict_.keys())})"

#------------------
# Selector
#------------------
class Selector(BaseEstimator, TransformerMixin):
    """Selects the features to be used in the model."""
    def __init__(self, features: List[str]) -> None:
        self.features: List[str] = features

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Selector":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.features]

    def __repr__(self) -> str:
        return f"Selector(features={self.features})"

#------------------
# JsonToDataFrame
#------------------
class JsonToDataFrame(BaseEstimator, TransformerMixin):
    """Transform JSON object to DataFrame."""
    def __init__(self) -> None:
        pass

    def fit(self, X: Any, y: Optional[Any] = None) -> "JsonToDataFrame":
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        return pd.DataFrame(X)

    def __repr__(self) -> str:
        return "JsonToDataFrame()"