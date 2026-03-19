# baseline_models.py

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from xgboost import XGBRegressor



def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


# ---------------------------------------------------------
# 2. TRAIN/TEST SPLIT
# ---------------------------------------------------------

def split_data(df, target, stratify_col=None, test_size=0.2):
    """
    Splits data into train/test sets.
    Optionally stratifies by a categorical column (e.g., condition).
    """
    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=df[stratify_col] if stratify_col else None
    )


# ---------------------------------------------------------
# 3. MODEL BUILDERS
# ---------------------------------------------------------

def build_linear_model(preprocessor):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])


def build_random_forest(preprocessor, n_estimators=300, max_depth=None):
    """
    Returns a pipeline: preprocessing + random forest regressor.
    """
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        ))
    ])
    return model



def build_xgboost_model(preprocessor):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        ))
    ])

# ---------------------------------------------------------
# 4. EVALUATION
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """
    Computes R², RMSE, and MAE for any trained model.
    """
    preds = model.predict(X_test)

    return {
        "r2": r2_score(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "mae": mean_absolute_error(y_test, preds)
    }
