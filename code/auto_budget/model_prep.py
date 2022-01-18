from auto_budget.content_encoder import content_encoder
# import date_encode
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

enc = OneHotEncoder(handle_unknown='ignore')
dectree = DecisionTreeClassifier()

total_day_seconds = 24*60*60

# de = date_encode.date_encode()

paymeth = {'Other' : 0, 'Debit Card' : 1, 'Deposit' : 2, 'VENMO' : 3, 'Service Charge': 4,
           'POS PURCHASE Non-PIN':5, 'POS PURCHASE with PIN':6}

class model_prep_select:
    def __init__(self, filename):
        self.data = pd.read_excel(filename)

    def encode_for_model(self, df):
        """
        Turn dates into sin/cos, time into seconds into sin/cos, encode categorical variables
        """
        # enc_df = df.copy(deep=True)
        enc_df = pd.DataFrame()
        # https://stackoverflow.com/questions/16236684/apply-pandas-function-to-column-to-create-multiple-new-columns
        # purchase_date_df = zip(df['Purchase Date'].map(de.date_each))
        # verification_date_df = zip(df['Verification Date'].map(de.date_each))

        # grab amount
        enc_df['Amount'] = df['Amount'].copy()

        # encode dates
        enc_df['PD_year'] = df['Purchase Date'].apply(lambda x: x.year)
        enc_df['sin_PD_month'] = np.sin(2 * np.pi * df['Purchase Date'].apply(lambda x: x.month) / 12)
        enc_df['cos_PD_month'] = np.cos(2 * np.pi * df['Purchase Date'].apply(lambda x: x.month) / 12)
        enc_df['sin_PD_day'] = np.sin(2 * np.pi * df['Purchase Date'].apply(lambda x: x.day) / 31)
        enc_df['cos_PD_day'] = np.cos(2 * np.pi * df['Purchase Date'].apply(lambda x: x.day) / 31)

        enc_df['VD_year'] = df['Verification Date'].apply(lambda x: x.year)
        enc_df['sin_VD_month'] = np.sin(2 * np.pi * df['Verification Date'].apply(lambda x: x.month) / 12)
        enc_df['cos_VD_month'] = np.cos(2 * np.pi * df['Verification Date'].apply(lambda x: x.month) / 12)
        enc_df['sin_VD_day'] = np.sin(2 * np.pi * df['Verification Date'].apply(lambda x: x.day) / 31)
        enc_df['cos_VD_day'] = np.cos(2 * np.pi * df['Verification Date'].apply(lambda x: x.day) / 31)

        # encode variables sklearn
        enc_df['Payment_Method_Cat'] = df['Payment_Method'].apply(lambda x: paymeth[x])
        paymeth_enc_df = pd.DataFrame(enc.fit_transform(enc_df[['Payment_Method_Cat']]).toarray())
        enc_df.drop(['Payment_Method_Cat'],axis=1,inplace=True)
        enc_df = enc_df.join(paymeth_enc_df)

        # encode words
        # ce = content_encoder(df.Content)
        # words_df = ce.onehot_encode_words()
        # enc_df = enc_df.join(words_df)

        # keep category
        enc_df['Category_Cat'] = df['Category_Cat'].copy()

        return enc_df


    def feature_selection(self, X, y):
        # initial fit
        model = SelectFromModel(estimator=DecisionTreeClassifier()).fit(X, y)
        status = model.get_support()
        X_new = model.transform(X)

        return X_new, status

    def select_best_model(self, model = KNeighborsClassifier(n_neighbors=12)): # DecisionTreeClassifier()):
        """runs split decision tree model and returns percent of matches"""
        # separate predictors and response
        prepped_df = self.encode_for_model(self.data)

        # reference dictionary
        ref = self.ref_dictionaries(self.data)

        # prepare X, y
        X = prepped_df.drop('Category_Cat', axis=1)
        y = prepped_df['Category_Cat'].copy()

        class_rates = {}

        # Feature selection
        # X_new, status = self.feature_selection(X, y)
        #
        # lis = np.array(X.columns)
        # cols_selected = lis[status]
        # print(cols_selected)
        cols_selected = list(X.columns)

        # test train split
        # X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.33, random_state=42)

        # for m in models:
        model.fit(X, y)
        # predictions = model.predict(X_test)
        # class_rates[m] = (sum(predictions == y_test.array) / len(predictions)) * 100

        return model, cols_selected, ref

    def ref_dictionaries(self, df):
        word_to_num_ref = df[['Category', 'Category_Cat']].groupby('Category').mean().to_dict()['Category_Cat'].items()
        num_to_word_ref = map(reversed, word_to_num_ref)

        return dict(word_to_num_ref), dict(num_to_word_ref)

    def ingest_to_data(self, df):
        # must drop columns first
        df = df[list(self.data.columns)].copy()

        # concatenate and upload
        final_df = pd.concat([df, self.data])
        final_df.to_excel('../data/CleanedStatements/new_data_check.xlsx', index=False)