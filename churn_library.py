# library doc string


# import libraries
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()



def import_data(pth: str) -> pd.DataFrame:
    """import data from a csv file

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
    return data_df


def add_chrun_column(df: pd.DataFrame) -> pd.DataFrame:
    """ add churn column to the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe containing the data
    """
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
        )
    return df



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
    
    plt.figure(figsize=(20,10)) 
    df["Churn"].hist()
    plt.savefig("./images/eda_churn.png")

    plt.figure(figsize=(20,10)) 
    df["Customer_Age"].hist()
    plt.savefig("./images/eda_customer_age.png")

    plt.figure(figsize=(20,10)) 
    df["Marital_Status"].value_counts("normalize").plot(kind="bar")
    plt.savefig("./images/eda_marital_status.png")

    plt.figure(figsize=(20,10))
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    plt.savefig("./images/eda_total_trans_ct.png")

    plt.figure(figsize=(20,10))
    df_only_numeric = df.select_dtypes(include=["float64", "int64"]).corr()
    sns.heatmap(df_only_numeric, annot=False, cmap="Dark2_r", linewidths = 2)
    plt.savefig("./images/eda_corr_heatmap.png")


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
        response: list,
) -> pd.DataFrame:
    """helper function to perform target encoding on given column

    Parameters
    ----------
    df : pd.DataFrame
        pandas dataframe containing the data
    category_lst : list
        list of columns to perform target encoding
    response : list
        list of new column names for target encoded columns

    Raises
    ------
    ValueError
        if required columns are not found in the dataframe

    Returns
    -------
    df : pd.DataFrame
        pandas dataframe containing the data with target encoded columns
    """

    for col in category_lst:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in DataFrame")

    for col, new_col in zip(category_lst, response):
        df[new_col] = target_encoding(df, col)

    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass
