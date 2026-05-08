"""Built-in sklearn datasets for the mini ML-engineering domain."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from sklearn import datasets as sklearn_datasets

from stemds.ml.tasks import MLEngineeringTask

TARGET_COLUMN = "target"


def load_sklearn_dataset(dataset_name: str) -> tuple[pd.DataFrame, str]:
    loaders: dict[str, Callable] = {
        "iris": sklearn_datasets.load_iris,
        "wine": sklearn_datasets.load_wine,
        "breast_cancer": sklearn_datasets.load_breast_cancer,
        "diabetes": sklearn_datasets.load_diabetes,
        "digits": sklearn_datasets.load_digits,
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unsupported sklearn dataset: {dataset_name}")

    bunch = loaders[dataset_name](as_frame=True)
    if getattr(bunch, "frame", None) is not None:
        dataframe = bunch.frame.copy()
        if TARGET_COLUMN not in dataframe.columns:
            dataframe[TARGET_COLUMN] = bunch.target
        return _clean_columns(dataframe), TARGET_COLUMN

    feature_names = getattr(bunch, "feature_names", None) or [f"feature_{index}" for index in range(bunch.data.shape[1])]
    dataframe = pd.DataFrame(bunch.data, columns=[str(name) for name in feature_names])
    dataframe[TARGET_COLUMN] = bunch.target
    return _clean_columns(dataframe), TARGET_COLUMN


def create_builtin_ml_tasks() -> list[MLEngineeringTask]:
    return [
        MLEngineeringTask(
            task_id="iris_accuracy",
            dataset_name="iris",
            target_name=TARGET_COLUMN,
            problem_type="classification",
            metric="accuracy",
            min_score=0.85,
            tags=["classification", "small_dataset", "multiclass"],
            description="Train a classifier for iris and report test accuracy.",
        ),
        MLEngineeringTask(
            task_id="iris_f1_macro",
            dataset_name="iris",
            target_name=TARGET_COLUMN,
            problem_type="classification",
            metric="f1_macro",
            min_score=0.85,
            tags=["classification", "small_dataset", "multiclass", "f1_macro"],
            description="Train a classifier for iris and report macro F1.",
        ),
        MLEngineeringTask(
            task_id="wine_accuracy",
            dataset_name="wine",
            target_name=TARGET_COLUMN,
            problem_type="classification",
            metric="accuracy",
            min_score=0.80,
            tags=["classification", "tabular", "multiclass"],
            description="Train a classifier for wine and report test accuracy.",
        ),
        MLEngineeringTask(
            task_id="breast_cancer_accuracy",
            dataset_name="breast_cancer",
            target_name=TARGET_COLUMN,
            problem_type="classification",
            metric="accuracy",
            min_score=0.90,
            tags=["classification", "tabular", "binary"],
            description="Train a classifier for breast cancer and report test accuracy.",
        ),
        MLEngineeringTask(
            task_id="breast_cancer_f1_macro",
            dataset_name="breast_cancer",
            target_name=TARGET_COLUMN,
            problem_type="classification",
            metric="f1_macro",
            min_score=0.90,
            tags=["classification", "tabular", "binary", "f1_macro"],
            description="Train a classifier for breast cancer and report macro F1.",
        ),
        MLEngineeringTask(
            task_id="diabetes_rmse",
            dataset_name="diabetes",
            target_name=TARGET_COLUMN,
            problem_type="regression",
            metric="rmse",
            tags=["regression", "tabular", "rmse"],
            description="Train a regressor for diabetes and report test RMSE.",
        ),
        MLEngineeringTask(
            task_id="diabetes_r2",
            dataset_name="diabetes",
            target_name=TARGET_COLUMN,
            problem_type="regression",
            metric="r2",
            tags=["regression", "tabular", "r2"],
            description="Train a regressor for diabetes and report test R2.",
        ),
        MLEngineeringTask(
            task_id="digits_accuracy",
            dataset_name="digits",
            target_name=TARGET_COLUMN,
            problem_type="classification",
            metric="accuracy",
            min_score=0.90,
            tags=["classification", "image_features", "multiclass"],
            description="Train a classifier for digits and report test accuracy.",
        ),
    ]


def _clean_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    cleaned.columns = [
        str(column).strip().replace(" ", "_").replace("/", "_per_").replace("-", "_")
        for column in cleaned.columns
    ]
    return cleaned
