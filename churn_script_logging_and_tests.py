import os
import logging
from typing import Callable

import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data: str):
    """test for import_data function

    Parameters
    ----------
    import_data : str
        path to the csv file

    Raises
    ------
    err : FileNotFoundError, AssertionError
        File not found error or assertion error
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


def test_eda(perform_eda: Callable):
    """test for perform_eda function

    Parameters
    ----------
    perform_eda : Callable
        function to perform exploratory data analysis

    Raises
    ------
    err: FileNotFoundError
        File not found error while readint dataframe
    err: ValueError 
        missing required columns while performing eda
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        df = cls.add_chrun_column(df.copy())
        
        cls.perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda: The file wasn't found")
        raise err
    except ValueError as err:
        logging.error("Testing perform_eda: %s", err)
        raise err



def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''


def test_train_models(train_models):
    '''
    test train_models
    '''


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
