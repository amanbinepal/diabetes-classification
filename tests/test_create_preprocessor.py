import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.functions.create_preprocessor import create_preprocessor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def test_create_preprocessor():
    numeric_feats = ["age", "income"]
    passthrough_feats = ["city"]

    preprocessor = create_preprocessor(numeric_feats, passthrough_feats)

    transformers = preprocessor.transformers

    scaler_name, scaler, scaler_cols = transformers[0]
    assert isinstance(scaler, StandardScaler), "Numeric features should use StandardScaler"
    assert scaler_cols == numeric_feats, "Numeric features not assigned correctly"

    passthrough_name, passthrough, passthrough_cols = transformers[1]
    assert passthrough == "passthrough", "Passthrough transformer not set"
    assert passthrough_cols == passthrough_feats, "Passthrough features not assigned correctly"