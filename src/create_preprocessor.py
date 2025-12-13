from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler

def create_preprocessor(numeric_feats, passthrough_feats):
    """
    Build a ColumnTransformer that scales numeric features and passes through other features.
    This function takes in list parameter and creates a scikit-learn ColumnTransformer that applies
    appropriate transformations to different feature types:
    - StandardScaler for numeric features 
    - passthrough for passthrough features

    Parameters
    ----------
    numeric_feats: list[str] 
        List of columns names for the numeric features
    passthrough_feats: list[str]
        List of columns names for the passthrough features

    Returns:
        An unfitted transformer to be used in a pipeline.
    
    Examples
    --------
    >>> preprocessor = create_preprocessor(numeric_feats, passthrough_feats)
    >>> preprocessor.fit(X_train)
    >>> X_transformed = preprocessor.transform(X_test)
    """
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_feats),
        ("passthrough", passthrough_feats)
    )
    return preprocessor

#from create_preprocessor import create_preprocessor
#preprocessor = create_preprocessor(numeric_feats=numeric_feats, passthrough_feats=passthrough_feats)