"""
Module to test churn_library.py

Author: Odin Eufracio
Date: Oct 2023
"""
import os
import logging
from typing import Callable, Optional
from collections import namedtuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

import churn_library as cls

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s"
)

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


def test_import(
        import_data: Callable[[str], pd.DataFrame]
):
    """test for import_data function

    Parameters
    ----------
    import_data : Callable[[str], pd.DataFrame]
        function to import data

    Raises
    ------
    err : FileNotFoundError
        File not found error while reading it dataframe
    err : AssertionError
        File does not have any rows or columns
    """

    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: the file does not have any rows or columns")
        raise err


def test_eda(
        perform_eda: Callable[[pd.DataFrame], None]
):
    """test for perform_eda function

    Parameters
    ----------
    perform_eda : Callable[[pd.DataFrame], None]
        function to perform exploratory data analysis

    Raises
    ------
    err: ValueError
        missing required columns while performing eda
    err: AssertionError
        Plots were not generated
    """
    try:
        df = cls.import_data("./data/bank_data.csv")

        perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except ValueError as err:
        logging.error("Testing perform_eda: %s", err)
        raise err

    try:
        assert os.path.exists("./images/eda_churn.png"), \
            "The file ./images/eda_churn.png was not generated"
        assert os.path.exists("./images/eda_customer_age.png"), \
            "The file ./images/eda_customer_age.png was not generated"
        assert os.path.exists("./images/eda_marital_status.png"), \
            "The file ./images/eda_marital_status.png was not generated"
        assert os.path.exists("./images/eda_marital_status.png"), \
            "The file ./images/eda_marital_status.png was not generated"
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: %s", err
        )
        raise err


def test_encoder_helper(
        encoder_helper: Callable[[pd.DataFrame, list, str], pd.DataFrame]
):
    """test for encoder_helper function

    Parameters
    ----------
    encoder_helper : Callable
        function to perform exploratory data analysis

    Raises
    ------
    err: ValueError
        missing required columns while performing encoding
    err: AssertionError
        new columns were not created or are not numeric
    """

    category_columns = [
        ("Gender", "Gender_Churn"),
        ("Education_Level", "Education_Level_Churn"),
        ("Marital_Status", "Marital_Status_Churn"),
        ("Income_Category", "Income_Category_Churn"),
        ("Card_Category", "Card_Category_Churn"),
    ]

    try:
        df = cls.import_data("./data/bank_data.csv")

        df = encoder_helper(
            df.copy(),
            category_columns,
            "Churn"
        )
        logging.info("Testing encoder_helper: SUCCESS")

    except ValueError as err:
        logging.error("Testing encoder_helper: %s", err)
        raise err

    try:
        for _, new_col in category_columns:
            assert new_col in df.columns, \
                f"The column {new_col} was not created"
            assert df[new_col].dtype == "float64", \
                f"The column {new_col} is not an integer"
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: %s", err
        )
        raise err



def test_perform_feature_engineering(
        perform_feature_engineering: Callable[
            [pd.DataFrame, list, list, Optional[str]],
            tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        ]
):
    """ test for perform_feature_engineering function

    Parameters
    ----------
    perform_feature_engineering : Callable
        function to perform feature engineering

    Raises
    ------
    err: ValueError
        missing required columns while performing feature engineering
    err: AssertionError
        missing required columns while performing feature engineering
    """

    category_columns = [
        ("Gender", "Gender_Churn"),
        ("Education_Level", "Education_Level_Churn"),
        ("Marital_Status", "Marital_Status_Churn"),
        ("Income_Category", "Income_Category_Churn"),
        ("Card_Category", "Card_Category_Churn"),
    ]

    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]

    try:
        df = cls.import_data("./data/bank_data.csv")

        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df.copy(),
            category_columns,
            keep_cols,
        )
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except ValueError as err:
        logging.error(
            "Testing perform_feature_engineering: %s", err
        )
        raise err

    try:
        assert not X_train.empty, \
            "The X_train dataframe is empty"
        assert not X_test.empty, \
            "The X_test dataframe is empty"
        assert not y_train.empty, \
            "The y_train dataframe is empty"
        assert not y_test.empty, \
            "The y_test dataframe is empty"
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: %s", err
        )
        raise err


def test_train_models(
        train_models: Callable[
            [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
            None
        ]
):
    """ test for train_models function

    Parameters
    ----------
    train_models : Callable
        function to train models

    Raises
    ------
    err: TypeError
        missing required columns while performing feature engineering
    err: AssertionError
        missing required columns while performing feature engineering
    """

    try:
        df = cls.import_data("./data/bank_data.csv")

        category_columns = [
            ("Gender", "Gender_Churn"),
            ("Education_Level", "Education_Level_Churn"),
            ("Marital_Status", "Marital_Status_Churn"),
            ("Income_Category", "Income_Category_Churn"),
            ("Card_Category", "Card_Category_Churn"),
        ]

        keep_cols = [
            "Customer_Age",
            "Dependent_count",
            "Months_on_book",
            "Total_Relationship_Count",
            "Months_Inactive_12_mon",
            "Contacts_Count_12_mon",
            "Credit_Limit",
            "Total_Revolving_Bal",
            "Avg_Open_To_Buy",
            "Total_Amt_Chng_Q4_Q1",
            "Total_Trans_Amt",
            "Total_Trans_Ct",
            "Total_Ct_Chng_Q4_Q1",
            "Avg_Utilization_Ratio",
            "Gender_Churn",
            "Education_Level_Churn",
            "Marital_Status_Churn",
            "Income_Category_Churn",
            "Card_Category_Churn",
        ]

        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df.copy(),
            category_columns,
            keep_cols,
        )

        train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except TypeError as err:
        logging.error(
            "Testing train_models: %s", err
        )
        raise err
    except AssertionError as err:
        logging.error(
            "Testing train_models: %s", err
        )
        raise err

    try:
        assert os.path.exists("./models/rfc_model.pkl"), \
            "The file ./models/rfc_model.pkl was not generated"
        assert os.path.exists("./models/lrc_model.pkl"), \
            "The file ./models/lrc_model.pkl was not generated"
        assert os.path.exists("./images/roc_curve.png"), \
            "The file ./images/roc_curve.png was not generated"
    except AssertionError as err:
        logging.error(
            "Testing train_models: %s", err
        )
        raise err


def test_feature_importance_plot(
        feature_importance_plot: Callable[
            [RandomForestClassifier, pd.DataFrame, str],
            None
        ]
):
    """ test for feature_importance_plot function

    Parameters
    ----------
    feature_importance_plot : Callable
        function to plot feature importance

    Raises
    ------
    err: TypeError
        missing required columns while performing feature engineering
    err: AssertionError
        missing required columns while performing feature engineering
    """
    try:
        df = cls.import_data("./data/bank_data.csv")

        category_columns = [
            ("Gender", "Gender_Churn"),
            ("Education_Level", "Education_Level_Churn"),
            ("Marital_Status", "Marital_Status_Churn"),
            ("Income_Category", "Income_Category_Churn"),
            ("Card_Category", "Card_Category_Churn"),
        ]

        keep_cols = [
            "Customer_Age",
            "Dependent_count",
            "Months_on_book",
            "Total_Relationship_Count",
            "Months_Inactive_12_mon",
            "Contacts_Count_12_mon",
            "Credit_Limit",
            "Total_Revolving_Bal",
            "Avg_Open_To_Buy",
            "Total_Amt_Chng_Q4_Q1",
            "Total_Trans_Amt",
            "Total_Trans_Ct",
            "Total_Ct_Chng_Q4_Q1",
            "Avg_Utilization_Ratio",
            "Gender_Churn",
            "Education_Level_Churn",
            "Marital_Status_Churn",
            "Income_Category_Churn",
            "Card_Category_Churn",
        ]

        _, X_test, _, _ = cls.perform_feature_engineering(
            df.copy(),
            category_columns,
            keep_cols,
        )

        rfc_model = joblib.load("./models/rfc_model.pkl")
        print(type(rfc_model))

        feature_importance_plot(
            rfc_model,
            X_test,
            "./images/feature_importance.png"
        )
        logging.info("Testing feature_importance_plot: SUCCESS")
    except TypeError as err:
        logging.error(
            "Testing feature_importance_plot: %s", err
        )
        raise err
    except AssertionError as err:
        logging.error(
            "Testing feature_importance_plot: %s", err
        )
        raise err

    try:
        assert os.path.exists("./images/feature_importance.png"), \
            "The file ./images/feature_importance.png was not generated"
    except AssertionError as err:
        logging.error(
            "Testing feature_importance_plot: %s", err
        )
        raise err


def test_classification_report_image(
        classification_report_image: Callable[[Responses], None]
):
    """ test for classification_report_image function

    Parameters
    ----------
    classification_report_image : Callable
        function to plot classification report

    Raises
    ------
    err: TypeError
        missing required columns while performing feature engineering
    err: AssertionError
        missing required columns while performing feature engineering
    """

    try:
        df = cls.import_data("./data/bank_data.csv")

        category_columns = [
            ("Gender", "Gender_Churn"),
            ("Education_Level", "Education_Level_Churn"),
            ("Marital_Status", "Marital_Status_Churn"),
            ("Income_Category", "Income_Category_Churn"),
            ("Card_Category", "Card_Category_Churn"),
        ]

        keep_cols = [
            "Customer_Age",
            "Dependent_count",
            "Months_on_book",
            "Total_Relationship_Count",
            "Months_Inactive_12_mon",
            "Contacts_Count_12_mon",
            "Credit_Limit",
            "Total_Revolving_Bal",
            "Avg_Open_To_Buy",
            "Total_Amt_Chng_Q4_Q1",
            "Total_Trans_Amt",
            "Total_Trans_Ct",
            "Total_Ct_Chng_Q4_Q1",
            "Avg_Utilization_Ratio",
            "Gender_Churn",
            "Education_Level_Churn",
            "Marital_Status_Churn",
            "Income_Category_Churn",
            "Card_Category_Churn",
        ]

        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df.copy(),
            category_columns,
            keep_cols,
        )

        rfc_model = joblib.load("./models/rfc_model.pkl")
        lrc_model = joblib.load("./models/lrc_model.pkl")

        y_train_preds_rfc = pd.Series(rfc_model.predict(X_train))
        y_test_preds_rfc = pd.Series(rfc_model.predict(X_test))

        y_train_preds_lrc = pd.Series(lrc_model.predict(X_train))
        y_test_preds_lrc = pd.Series(lrc_model.predict(X_test))

        classification_report_image(Responses(
            y_train,
            y_test,
            y_train_preds_lrc,
            y_train_preds_rfc,
            y_test_preds_lrc,
            y_test_preds_rfc
        ))

        logging.info("Testing classification_report_image: SUCCESS")
    except TypeError as err:
        logging.error(
            "Testing classification_report_image: %s", err
        )
        raise err
    except AssertionError as err:
        logging.error(
            "Testing classification_report_image: %s", err
        )
        raise err

    try:
        assert os.path.exists("./images/classification_report.png"), \
            "The file ./images/classification_report_train.png was not generated"
    except AssertionError as err:
        logging.error(
            "Testing classification_report_image: %s", err
        )
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
    test_feature_importance_plot(cls.feature_importance_plot)
    test_classification_report_image(cls.classification_report_image)
