# https://www.kaggle.com/shakedzy/alone-in-the-woods-using-theil-s-u-for-survival
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import math
from collections import Counter
import scipy.stats as ss
import sklearn.preprocessing as sp
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from subprocess import check_output
from sklearn import metrics
from openpyxl import load_workbook

labelencoder = LabelEncoder()
enc = OneHotEncoder(handle_unknown='ignore')
rf = RandomForestClassifier()
dectree = DecisionTreeClassifier()
paymeth = {'Other' : 0, 'Debit Card' : 1, 'Deposit' : 2, 'VENMO' : 3, 'Service Charge': 4}

# Dataframe Visual Settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 400)


def PIN_col(x):
    if "POS PURCHASE Non-PIN" in x:
        return "POS PURCHASE Non-PIN"
    elif "POS PURCHASE with PIN" in x:
        return "POS PURCHASE with PIN"
    elif "VENMO" in x:
        return "VENMO"
    else:
        return "Other"


def remove_payment_method(x):
    if "POS PURCHASE Non-PIN" in x:
        return x.split("POS PURCHASE Non-PIN")[1]
    elif "POS PURCHASE with PIN" in x:
        return x.split("POS PURCHASE with PIN")[1]
    else:
        return x


def middle_words(x):
    if (len(x) > 2):
        return x[1:-1]
    else:
        return "N/A"

# def time_midnight(x):
#     if type(x) == PT:
#         return x.time()
#     else:
#         return x


def yield_seconds(x):
    #     x = time_midnight(x)
    #     print(type(x))
    return dt.timedelta(hours=x.hour, minutes=x.minute, seconds=x.second).total_seconds()


def no_space(x):
    if x[-1] == ' ':
        return x[:-1]
    else:
        return x


total_day_seconds = 24*60*60


def date_parse_encode(sr):
    pass


def encode_data(df):
    """
    Turn dates into sin/cos, time into seconds into sin/cos, encode categorical variables
    """
    df['PD_year'] = df['Purchase Date'].apply(lambda x: x.year)
    df['sin_PD_month'] = np.sin(2 * np.pi * df['Purchase Date'].apply(lambda x: x.month) / 12)
    df['cos_PD_month'] = np.cos(2 * np.pi * df['Purchase Date'].apply(lambda x: x.month) / 12)
    df['sin_PD_day'] = np.sin(2 * np.pi * df['Purchase Date'].apply(lambda x: x.day) / 31)
    df['cos_PD_day'] = np.cos(2 * np.pi * df['Purchase Date'].apply(lambda x: x.day) / 31)
    df['PT_total_seconds'] = df['Purchase Time'].apply(yield_seconds)
    df['sin_PT_total_seconds'] = np.sin(2 * np.pi * df['PT_total_seconds'] / total_day_seconds)
    df['cos_PT_total_seconds'] = np.cos(2 * np.pi * df['PT_total_seconds'] / total_day_seconds)
    df['VD_year'] = df['Verification Date'].apply(lambda x: x.year)
    df['sin_VD_month'] = np.sin(2 * np.pi * df['Verification Date'].apply(lambda x: x.month) / 12)
    df['cos_VD_month'] = np.cos(2 * np.pi * df['Verification Date'].apply(lambda x: x.month) / 12)
    df['sin_VD_day'] = np.sin(2 * np.pi * df['Verification Date'].apply(lambda x: x.day) / 31)
    df['cos_VD_day'] = np.cos(2 * np.pi * df['Verification Date'].apply(lambda x: x.day) / 31)

    # encode variables sklearn
    df['Payment_Method_Cat'] = df['Payment_Method'].apply(lambda x: paymeth[x])
    enc_df = pd.DataFrame(enc.fit_transform(df[['Payment_Method_Cat']]).toarray())
    df = df.join(enc_df)
    df[2] = 0
    df[3] = 0

    # df['First_Word_Cat'] = labelencoder.fit_transform(df['First_Word'].apply(lambda x: str(x)))
    # only do when training/testing the model, not with real data
    # df['Category_Cat'] = labelencoder.fit_transform(df['Category'])
    # print("encode_data: ", df.index)

    return df

def ref_dictionaries(df):
    word_to_num_ref = df[['Category', 'Category_Cat']].groupby('Category').mean().to_dict()['Category_Cat'].items()
    num_to_word_ref = map(reversed, word_to_num_ref)

    return dict(word_to_num_ref), dict(num_to_word_ref)

def decision_tree_new_statements(df, new_df):
    """
    Actually use the random forest to gather predictions, decode the predictions
    and place predictions into pd series
    """
    X = df.drop(['Purchase Date','Purchase Time','Payment_Method','Verification Date', 'Content',
                 'Category','PT_total_seconds','Payment_Method_Cat', 'Category_Cat'], axis=1)
                 # # # DROP THE FW_CAT COLUMN ON THE MAIN TABLE ! ! ! you want the original cat_cat column
    new_X = new_df.drop(['Purchase Date','Purchase Time','Payment_Method','Verification Date',
                         'Content', 'Category','PT_total_seconds','Payment_Method_Cat', 'Category_Pred',
                         'Cat_Self_Pred', 'Cat_Pred_Words', 'Self_Pred_Nums'], axis=1)
    y = df['Category_Cat'].copy() #Category Cat is numbers

    # train model for prediction
    dectree.fit(X, y)
    new_predictions = dectree.predict(new_X)

    return pd.Series(new_predictions)


def self_categorization(df, ref):
    """
    categorize the transactions as they actually should be
    """

    # results of the decision tree model
    df['Cat_Pred_Words'] = df['Category_Pred'].apply(lambda x: ref[1][x])

    # quickly self-categorizing based on my own input
    df['Cat_Self_Pred'] = categorize(df) # let's see if this works, or if my approach needs a change of direction (it does)

    # user inputs for 'Cat_Self_Pred' as 'Default' | NEXT ! build error checking for non-values within dict
    for idx in df.index[df['Cat_Self_Pred'] == 'Default']:
        print(df.loc[idx, ['Content', 'Amount', 'Cat_Pred_Words']])
        ans = input()
        if ans == 'y':
            df.loc[idx, 'Cat_Self_Pred'] = df.loc[idx, 'Cat_Pred_Words'] # assign to decision tree's answer
        else:
            df.loc[idx, 'Cat_Self_Pred'] = ans

    df['Category_Cat'] = df['Cat_Self_Pred'].apply(lambda x: ref[0][x])

    return df["Cat_Self_Pred"]

def upload_to_excel_budget(df):
    # Upload excel to 2021/2/or whatever budget spreadsheet
    new_summary = df.groupby('Cat_Self_Pred').agg({'Amount':'sum'})
    new_summary.loc[new_summary['Amount'] < 0, 'Portion_of'] = new_summary[new_summary['Amount'] < 0].apply(lambda x: 100 * x / float(x.sum()))['Amount']
    new_summary.loc[new_summary['Amount'] > 0, 'Portion_of'] = new_summary[new_summary['Amount'] > 0].apply(lambda x: 100 * x / float(x.sum()))['Amount']
    new_summary['Portion_of'] = new_summary['Portion_of'].apply(lambda x: "{0:.2f}%".format(x))
    new_summary['Amount'] = new_summary['Amount'].apply(lambda x: "${:,.2f}".format(x))

    # file_to_upload = 'tests/Budget 2021.xlsx'
    excel_book = load_workbook('tests/Budget 2021.xlsx')
    with pd.ExcelWriter('tests/Budget 2021.xlsx', engine='openpyxl') as writer:
        writer.book = excel_book
        writer.sheets = {ws.title:ws for ws in excel_book.worksheets}
        new_summary.to_excel(writer, "May1", index=True)
        writer.save()

def model_statistics_performance(df):
    """
    For the purpose of training the model and seeing its effectiveness
    """
    # df['Category_Pred_Words'] = df['Category_Pred'].apply(lambda x: ref[0][x])

    # new_df['Cat_Self_Pred_Num'] = new_df['Cat_Self_Pred'].apply(lambda x: word_to_num_ref[x])

    # y_test = new_df['Cat_Self_Pred_Num'].to_numpy() # the real categories in numbers
    # y_pred = new_predictions # the predicted categories

    # # some comparison statistics
    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    pass

def decode_data(df):
    """
    I don't know why this is necessary actually, because in rcf I'd only be using necessary vars
    And I can just keep the encoded vars
    :param df:
    :return:
    """
    pass

def clean_bank_statement_file(filename):
    """
    Extract, modify & clean bank statement raw file for neater format
    """
    # Cleaning data
    # csv_file = '\Bank Statements\'' + filename
    df = pd.read_csv(filename, usecols=['Posting Date', 'Effective Date','Transaction Type', 'Amount',
                                                                        'Description', 'Type',
                                                                        'Extended Description'])
    df['Date'] = pd.to_datetime(df['Posting Date'])
    df['Purchase Date'] = pd.to_datetime(df['Effective Date'])
    df['Content'] = df['Description'] + ' ' + df['Extended Description']
    df['Category'] = 'Default'
    df['Payment_Method'] = df['Type'].copy()
    df['Purchase Time'] = dt.time(0, 0, 0)
    df['Verification Date'] = df['Date'].copy()
    df.drop(df[df['Content'].str.contains("Transfer", na=False)].index, inplace=True)
    df['Category_Pred'] = -1
    df['Cat_Pred_Words'] = 'Un_catted'
    df['Cat_Self_Pred'] = 'Un_catted'
    df['Self_Pred_Nums'] = -1
    df.drop(
        ['Date', 'Posting Date', 'Effective Date', 'Transaction Type', 'Description', 'Type', 'Extended Description'],
        axis=1, inplace=True)
    df.sort_values(by=['Purchase Date', 'Purchase Time'], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[['Purchase Date', 'Purchase Time', 'Payment_Method', 'Amount', 'Verification Date', 'Content', 'Category',
             'Category_Pred', 'Cat_Pred_Words', 'Cat_Self_Pred', 'Self_Pred_Nums']].copy()
    return df


def categorize(df):
    """
    For common transactions, categorize these early, and use to train a model
    """
    # Easier Categorization
    # Also categorizing the actual dataframe, but not a huge problem
    df.loc[df['Content'].str.contains("UBER EATS", na=False), 'Category'] = 'Dining Out'
    df.loc[df['Content'].str.contains("HARRIS", na=False), 'Category'] = 'Groceries'
    df.loc[df['Content'].str.contains("GIANT", na=False), 'Category'] = 'Groceries'
    df.loc[df['Content'].str.contains("USAA", na=False), 'Category'] = 'USAA Insurance'
    df.loc[df['Content'].str.contains("Accenture", na=False), 'Category'] = 'Pay Check'
    df.loc[df['Content'].str.contains("XSPORT", na=False), 'Category'] = 'Gym'
    df.loc[df['Content'].str.contains("DISTRICT MARTIAL ARTS", na=False), 'Category'] = 'Gym'
    df.loc[df['Content'].str.contains("PARKING", na=False), 'Category'] = 'Tolls/Uber/Metro/Parking'
    df.loc[df['Content'].str.contains("NAZRET", na=False), 'Category'] = 'Dining Out'
    df.loc[df['Content'].str.contains("TAJ OF INDIA", na=False), 'Category'] = 'Dining Out'
    df.loc[df['Content'].str.contains("DCPILLAR", na=False), 'Category'] = 'Tithe'
    df.loc[df['Content'].str.contains("GOOGLE", na=False), 'Category'] = 'Entertainment'
    df.loc[df['Content'].str.contains("VENMO/CASHOUT", na=False), 'Category'] = 'Venmo Extra'
    df.loc[df['Content'].str.contains("CITGO", na=False), 'Category'] = 'Gas'
    df.loc[df['Content'].str.contains("SHELL", na=False), 'Category'] = 'Gas'
    df.loc[df['Content'].str.contains("PUPATELLA", na=False), 'Category'] = 'Dining Out'
    df.loc[df['Content'].str.contains("GOOD COMPANY DONUT", na=False), 'Category'] = 'Dining Out'
    df.loc[df['Content'].str.contains("STARBUCKS", na=False), 'Category'] = 'Dining Out'
    df.loc[df['Content'].str.contains("UBER TRIP", na=False), 'Category'] = 'Tolls/Uber/Metro/Parking'
    df.loc[df['Content'].str.contains("VERIZON", na=False), 'Category'] = 'Utilities'
    df.loc[df['Content'].str.contains("WASHINGTON GAS", na=False), 'Category'] = 'Utilities'
    df.loc[df['Content'].str.contains("ENERGY", na=False), 'Category'] = 'Utilities'
    df.loc[df['Content'].str.contains("TOM COLEMAN", na=False), 'Category'] = 'Phone'
    df.loc[df['Content'].str.contains("STDNT LOAN", na=False), 'Category'] = 'Student Loans'
    df.loc[(df['Content'].str.contains("VENMO/PAYMENTWALTER COLEMAN Default", na=False)) &
           (df['Amount'] == -845), 'Category'] = 'Rent'
    df.loc[df['Content'].str.contains("Margaret Coleman", na=False), 'Category'] = 'Extra'
    df.loc[df['Content'].str.contains("Person-to-Person TransferPAYPAL", na=False), 'Category'] = 'Extra'
    df.loc[df['Content'].str.contains("Tortas y Tacos", na=False), 'Category'] = 'Dining Out'
    df.loc[df['Content'].str.contains("CROWNE PLAZA", na=False), 'Category'] = 'Dining Out'
    df.loc[df['Content'].str.contains("Emmaus Family Couns", na=False), 'Category'] = 'Medical'
    df.loc[df['Content'].str.contains("ADVANCED HEALTH CARE", na=False), 'Category'] = 'Medical'
    df.loc[df['Content'].str.contains("AMZN Mktp", na=False), 'Category'] = 'Misc'
    df.loc[df['Content'].str.contains("Amazon web services", na=False), 'Category'] = 'Misc'
    df.loc[df['Content'].str.contains("ALDI", na=False), 'Category'] = 'Groceries'
    df.loc[df['Content'].str.contains("FOOD LION", na=False), 'Category'] = 'Groceries'
    df.loc[df['Content'].str.contains("Audible", na=False), 'Category'] = 'Entertainment'
    df.loc[df['Content'].str.contains("PIZZA", na=False), 'Category'] = 'Dining Out'
    df.loc[df['Content'].str.contains("CROWNE PLAZA", na=False), 'Category'] = 'Dining Out'
    # set positive defaults to Misc
    df.loc[(df['Category'] == 'Default') & (df['Amount'] > 0), 'Category'] = 'Extra'
    df.loc[df['Content'].str.contains("Pizza", na=False), 'Category'] = 'Dining Out'
    df.loc[df['Content'].str.contains("Amzn", na=False), 'Category'] = 'Misc'
    df.loc[df['Content'].str.contains("Pollo", na=False), 'Category'] = 'Dining Out'
    df.loc[df['Content'].str.contains("VZ WIRELESS", na=False), 'Category'] = 'Phone'
    df.loc[df['Content'].str.contains("PARKMOBILE", na=False), 'Category'] = 'Tolls/Uber/Metro/Parking'
    categories = pd.Series(df.Category)
    return categories


def create_grand_file(directory):
    """
    Extract based on string file names
    """
    grand_df = pd.DataFrame()
    temp_df = pd.DataFrame()

    # Get list of files
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            temp_df = clean_bank_statement_file(filename)
            temp_df = categorize(temp_df)
            grand_df = pd.concat([grand_df, temp_df], ignore_index=True)

    return grand_df

def dashboard(df, new_df):
    #Monthly groceries (all year)
    # Aggregate Groceries transactions by month x is month, y is amount-spent-on-groceries-that-month
    # .plot(restaurants, month_agg)
    # .plot(groceries, month_agg)
    # .plot(total, month_agg)
    # .show()
    #Monthly restaurants (all year)
    #Monthly total (saved) (all year)
    #This month pie
    # A pie chart of the purchases this month
    #24-hour spending
    #day of week spending
    pass