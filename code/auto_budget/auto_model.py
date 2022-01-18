import numpy as np
from auto_budget.model_prep import model_prep_select
import pandas as pd
import datetime as dt
from openpyxl import load_workbook
import os
# mod = model_prep_select('')

class budget_categorizer:
    def __init__(self, new_path, old_path):
        self.new_data = self.clean_bank_statement_file(new_path)
        self.model = model_prep_select(old_path)
        # clean file and format

    def clean_bank_statement_file(self, filename):
        """
        Extract, modify & clean bank statement raw file for neater format
        """
        # Cleaning data
        # csv_file = '\Bank Statements\'' + filename
        df = pd.read_csv(filename, usecols=['Posting Date', 'Effective Date', 'Transaction Type', 'Amount',
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
        df['Category_Cat'] = -1
        df.drop(
            ['Date', 'Posting Date', 'Effective Date', 'Transaction Type', 'Description', 'Type',
             'Extended Description'],
            axis=1, inplace=True)
        df.sort_values(by=['Purchase Date', 'Purchase Time'], ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = df[
            ['Purchase Date', 'Purchase Time', 'Payment_Method', 'Amount', 'Verification Date', 'Content', 'Category',
             'Category_Pred', 'Cat_Pred_Words', 'Cat_Self_Pred', 'Category_Cat']].copy()
        return df

    def predict_with_model(self):
        """Use best selected model to now encode the new transactions"""
        # encode new data file
        new_data_encoded = self.model.encode_for_model(self.new_data)

        # find best model given historic data
        model, cols, ref = self.model.select_best_model()

        # print(cols)

        # remove columns not in model
        new_data_encoded = new_data_encoded[[x for x in list(new_data_encoded.columns) if x in cols]].copy()

        print(new_data_encoded.columns)

        cols_in_model = [x for x in cols if x not in list(new_data_encoded.columns)]

        # add zeros for not available columns
        df_to_append = pd.DataFrame(0, index=np.arange(len(new_data_encoded)), columns = cols_in_model)
        new_data_encoded = new_data_encoded.join(df_to_append)

        print(new_data_encoded.columns)

        # gather predictions
        predictions = model.predict(new_data_encoded)
        pred = pd.Series(predictions).apply(lambda x: ref[1][x])

        self.new_data['Category_Pred'] = predictions

        self.self_categorization(ref)

        # for reviewing model, will come back to this
        # return pd.DataFrame(data = {'memo': self.new_data.Content, 'prediction': pred})

    def self_categorization(self, ref):
        """
        categorize the transactions as they actually should be
        """
        # results of the decision tree model
        self.new_data['Cat_Pred_Words'] = self.new_data['Category_Pred'].apply(lambda x: ref[1][x])

        # quickly self-categorizing based on my own input
        self.new_data['Cat_Self_Pred'] = self.categorize(self.new_data)

        # user inputs for 'Cat_Self_Pred' as 'Default' | NEXT ! build error checking for non-values within dict
        was_correct = 0
        for idx in self.new_data.index[self.new_data['Cat_Self_Pred'] == 'Default']:
            print(self.new_data.loc[idx, ['Content', 'Amount', 'Cat_Pred_Words']])
            ans = input()
            if ans == 'y':
                self.new_data.loc[idx, 'Cat_Self_Pred'] = self.new_data.loc[idx, 'Cat_Pred_Words']
                was_correct = was_correct + 1
            else:
                self.new_data.loc[idx, 'Cat_Self_Pred'] = ans

        # print accuracy
        accuracy = (was_correct/len(self.new_data)) * 100
        print("Model Accuracy: {0:.0f}%".format(accuracy))

        # adjust the new data category
        self.new_data['Category'] = self.new_data['Cat_Self_Pred'].copy()
        self.new_data['Category_Cat'] = self.new_data['Cat_Self_Pred'].apply(lambda x: ref[0][x])

        # ask whether to upload new info
        new_summary = self.new_data.groupby('Cat_Self_Pred').agg({'Amount':'sum'})
        print(new_summary)
        print("Proceed? Y/n \n")
        ans = input()
        if ans == "Y" or ans == "y":
            self.model.ingest_to_data(self.new_data)
            self.upload_to_excel_budget(new_summary)
        else:
            print("cat rejected, redo")
            self.self_categorization(ref)

    def upload_to_excel_budget(self, new_summary):
        # Upload excel to budget spreadsheet
        # new_summary = self.new_data.groupby('Cat_Self_Pred').agg({'Amount':'sum'})
        new_summary.loc[new_summary['Amount'] < 0, 'Portion_of'] = new_summary[new_summary['Amount'] < 0].apply(lambda x: 100 * x / float(x.sum()))['Amount']
        new_summary.loc[new_summary['Amount'] > 0, 'Portion_of'] = new_summary[new_summary['Amount'] > 0].apply(lambda x: 100 * x / float(x.sum()))['Amount']
        new_summary['Portion_of'] = new_summary['Portion_of'].apply(lambda x: "{0:.2f}%".format(x))
        new_summary['Amount'] = new_summary['Amount'].apply(lambda x: "${:,.2f}".format(x))

        # open budget file and create new sheet
        # cwd = os.getcwd()
        filename = '../../../Documents/Finances/Budget 2022 copy.xlsx'
        excel_book = load_workbook(filename)
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            writer.book = excel_book
            writer.sheets = {ws.title:ws for ws in excel_book.worksheets}
            new_summary.to_excel(writer, "Jan_test", index=False) # note to change this later
            writer.save()


    def categorize(self, df):
        """
        For common transactions, categorize these early, and use to train a model
        """
        # Easier Categorization
        # Also categorizing the actual dataframe, but not a huge problem
        df = df.copy(deep=True)
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