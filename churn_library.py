"""
Module that implements the functions to perform EDA, feature engineering,
model training and model evaluation

Author: Odin Eufracio
Date:   Oct 2023
"""


# import libraries
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import joblib

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sns.set()

Responses = namedtuple(
    'Responses',
    [
        'y_train',
        'y_test',
        'y_train_preds_lr',
        'y_train_preds_rf',
        'y_test_preds_lr',
        'y_test_preds_rf'
    ]
)


def import_data(pth: str) -> pd.DataFrame:
    """import data from a csv file the Churn column is added to the dataframe

    Parameters
    ----------
    pth : str
        path to the csv file

    Returns
    -------
    data_df : pd.DataFrame
        pandas dataframe containing the data
    """
    data_df = pd.read_csv(pth)

    data_df["Churn"] = data_df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    return data_df


def perform_eda(df: pd.DataFrame) -> None:
    """Perform Exploratory Data Analysis and save figures to ./images folder

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe containing the data

    Raises
    ------
    ValueError
        if required columns are not found in the dataframe
    """

    required_columns = [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        "Total_Trans_Ct",
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in DataFrame")

    if not os.path.exists("images"):
        os.mkdir("images")

    plt.figure(figsize=(20, 10))
    df["Churn"].hist()
    plt.savefig("./images/eda_churn.png")

    plt.figure(figsize=(20, 10))
    df["Customer_Age"].hist()
    plt.savefig("./images/eda_customer_age.png")

    plt.figure(figsize=(20, 10))
    df["Marital_Status"].value_counts("normalize").plot(kind="bar")
    plt.savefig("./images/eda_marital_status.png")

    plt.figure(figsize=(20, 10))
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    plt.savefig("./images/eda_total_trans_ct.png")

    plt.figure(figsize=(20, 10))
    df_only_numeric = df.select_dtypes(include=["float64", "int64"]).corr()
    sns.heatmap(df_only_numeric, annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig("./images/eda_corr_heatmap.png")
    plt.clf()
    plt.close()


def target_encoding(
        df: pd.DataFrame,
        column_name: str,
        target_name: str = 'Churn',
) -> pd.Series:
    """perform target encoding on given column

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe containing the data
    column_name : str
        column name to perform target encoding
    target_name : str
        target column name

    Returns
    -------
    df : pd.Series
        pandas series containing the target encoded column
    """

    encoding = df.groupby(column_name)[target_name].mean()
    return df[column_name].map(encoding)


def encoder_helper(
        df: pd.DataFrame,
        category_lst: list,
        response: str = "Churn",
) -> pd.DataFrame:
    """helper function to perform target encoding on given column

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe containing the data
    category_lst : list
        list of tuples containing  (categorial column name,  new column name)
    response : str
        response column name (default is "Churn")

    Raises
    ------
    ValueError
        if required columns are not found in the dataframe

    Returns
    -------
    df : pd.DataFrame
        pandas dataframe containing the data with target encoded columns
    """

    for col, _ in category_lst:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in DataFrame")

    if response not in df.columns:
        raise ValueError(
            f"Expected column '{response}' not found in DataFrame")

    for col, new_col in category_lst:
        df[new_col] = target_encoding(df, col, response)

    return df


def perform_feature_engineering(
        df: pd.DataFrame,
        categorial_columns: list,
        keep_cols: list,
        response: str = "Churn",
) -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
    """perform feature engineering on the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe containing the data
    categorial_columns : list
        list of tuples containing  (categorial column name,  new column name)
    keep_cols : list
        list of columns to keep in the dataframe
    response : str
        response column name (default is "Churn")

    Returns
    -------
    X_train : pd.DataFrame
        pandas dataframe containing the training data
    X_test : pd.DataFrame
        pandas dataframe containing the testing data
    y_train : pd.Series
        pandas series containing the training response data
    y_test : pd.Series
        pandas series containing the testing response data

    Raises
    ------
    ValueError
        if required columns are not found in the dataframe
    """

    try:
        df = encoder_helper(
            df.copy(),
            categorial_columns,
            response
        )
    except ValueError as err:
        raise err

    X = df[keep_cols]
    y = df[response]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )
    return X_train, X_test, y_train, y_test


def classification_report_image(
        responses: Responses
) -> None:
    """plot classification report and save to ./images/classification_report.png

    Parameters
    ----------
    responses : Responses
        namedtuple containing the responses, all of them are pandas Series:
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf

    Returns
    -------
    None

    Raises
    ------
    TypeError
        if any of the input parameters are not of the expected type
    AssertionError
        if any of the input parameters are empty
    """

    if not isinstance(responses.y_train, pd.Series):
        raise TypeError("y_train should be a pandas Series")
    if not isinstance(responses.y_test, pd.Series):
        raise TypeError("y_test should be a pandas Series")
    if not isinstance(responses.y_train_preds_lr, pd.Series):
        raise TypeError("y_train_preds_lr should be a pandas Series")
    if not isinstance(responses.y_train_preds_rf, pd.Series):
        raise TypeError("y_train_preds_rf should be a pandas Series")
    if not isinstance(responses.y_test_preds_lr, pd.Series):
        raise TypeError("y_test_preds_lr should be a pandas Series")
    if not isinstance(responses.y_test_preds_rf, pd.Series):
        raise TypeError("y_test_preds_rf should be a pandas Series")

    assert not responses.y_train.empty, \
        "y_train should not be empty"
    assert not responses.y_test.empty, \
        "y_test should not be empty"
    assert not responses.y_train_preds_lr.empty, \
        "y_train_preds_lr should not be empty"
    assert not responses.y_train_preds_rf.empty, \
        "y_train_preds_rf should not be empty"
    assert not responses.y_test_preds_lr.empty, \
        "y_test_preds_lr should not be empty"
    assert not responses.y_test_preds_rf.empty, \
        "y_test_preds_rf should not be empty"

    plt.figure(figsize=(8, 10))

    plt.text(
        0.01,
        1.5,
        str("Logistic Regression Train"),
        {"fontsize": 10},
        fontproperties="monospace"
    )
    plt.text(
        0.01,
        1.3,
        str(classification_report(responses.y_train, responses.y_train_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace"
    )

    plt.text(
        0.01,
        1.2,
        str("Logistic Regression Test"),
        {"fontsize": 10},
        fontproperties="monospace"
    )
    plt.text(
        0.01,
        1.0,
        str(classification_report(responses.y_test, responses.y_test_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace"
    )

    plt.text(
        0.01,
        0.9,
        str("Random Forest Train"),
        {"fontsize": 10},
        fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(responses.y_train, responses.y_train_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace"
    )

    plt.text(
        0.01,
        0.6,
        str("Random Forest Test"),
        {"fontsize": 10},
        fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.4,
        str(classification_report(responses.y_test, responses.y_test_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace"
    )

    plt.axis("off")
    plt.savefig(
        "./images/classification_report.png",
        bbox_inches="tight"
    )
    plt.clf()
    plt.close()


def feature_importance_plot(
        model: RandomForestClassifier,
        X_data: pd.DataFrame,
        output_pth: str
) -> None:
    """plot feature importance and save to output_pth

    Parameters
    ----------
    model : RandomForestClassifier
        trained random forest classifier model
    X_data : pd.DataFrame
        pandas dataframe containing the data
    output_pth : str
        path to save the plot

    Returns
    -------
    None
    """

    if not isinstance(model, RandomForestClassifier):
        raise TypeError("model should be a RandomForestClassifier")

    assert not X_data.columns.empty, \
        "X_data columns should not be empty"
    assert len(X_data.columns) == len(model.feature_importances_), \
        "X_data columns should be equal to model feature importances"

    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.clf()
    plt.close()


def train_models(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
) -> None:
    """train models and save roc curve to ../images/roc_curve.png
     and models to ./models/rfc_model.pkl and ./models/lrc_model.pkl

    Parameters
    ----------
    X_train : pd.DataFrame
        pandas dataframe containing the training data
    X_test : pd.DataFrame
        pandas dataframe containing the testing data
    y_train : pd.Series
        pandas series containing the training response data
    y_test : pd.Series
        pandas series containing the testing response data

    Returns
    -------
    None

    Raises
    ------
    TypeError
        if any of the input parameters are not of the expected type
    AssertionError
        if any of the input parameters are empty
    """

    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train should be a pandas DataFrame")
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test should be a pandas DataFrame")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train should be a pandas Series")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test should be a pandas Series")

    assert not X_train.empty, "X_train should not be empty"
    assert not X_test.empty, "X_test should not be empty"
    assert not y_train.empty, "y_train should not be empty"
    assert not y_test.empty, "y_test should not be empty"

    # Random Forest Classifier
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"]
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # Logistic Regression Classifier
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)
    lrc.fit(X_train, y_train)

    # Save models
    if not os.path.exists("models"):
        os.mkdir("models")
    with open("./models/rfc_model.pkl", "wb") as model_file:
        joblib.dump(cv_rfc.best_estimator_, model_file)
    with open("./models/lrc_model.pkl", "wb") as model_file:
        joblib.dump(lrc, model_file)

    # Plotting ROC Curve
    lrc_plot = plot_roc_curve(
        lrc,
        X_test,
        y_test,
    )

    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        **{"alpha": 0.8}
    )

    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("./images/roc_curve.png")
    plt.clf()
    plt.close()
