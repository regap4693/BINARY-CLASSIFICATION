import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from process_data.DataFrameImputer import DataFrameImputer
from process_data.ImBalance import ImBalance
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

class Preprocess(ImBalance):
    """Preprocess the source data before feeding the model
    """
    def __init__(self,type = None):
        super(Preprocess,self).__init__(type)
        self.mis_col = []
        self.train_cols = []
    
    def missing_values_table(self, df):
        """returns dataframe with missing information
        Args:
         df: input dataframe
        """
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum()/len(df)
    
        
        mis_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        mis_table_columns = mis_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        
        mis_table_columns = mis_table_columns[
        mis_table_columns.iloc[:,1] != 0].sort_values(
              '% of Total Values', ascending=False).round(1)

        print ( "There are " + str(mis_table_columns.shape[0]) + " columns that have missing values.")
        return mis_table_columns
    
    
    def readPreprocess(self, X):
        """Many steps performed here as part of preprocessing
        1. read the data
        2. replace some invalid values with np.nan
        3. perform label encoding
        4. Do below for each column with missing values eg.colA:
            i) use DataImputer to impute missing values
            ii) create a new column colA_miss for a column say colA. It stores value 1 if value is missing in colA and 0 otherwise.
                Basically this new column marks the record where the existing column has missing values.

        Args:
         source: sourcefile path
        Output:
         Preprocessed data
        """
        
        X['CODE_GENDER'].replace('XNA',np.nan, inplace=True)
        X['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
        X['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
        le = LabelEncoder()
        le_count = 0
        # Iterate through the columns
        for col in X:
            if X[col].dtype == 'object':
                    # If 2 or fewer unique categories
                    if len(list(X[col].unique())) <= 2:
                        # Train on the training data
                        le.fit(X[col])
                        # Transform both training and testing data
                        X[col] = le.transform(X[col])
                        # Keep track of how many columns were label encoded
                        le_count += 1
        print('%d columns were label encoded.' % le_count)
        X = pd.get_dummies(X)
        if len(self.mis_col)==0:
            missing_values = self.missing_values_table(X)
            self.mis_col=list(missing_values.index)
        
        Mis = X[self.mis_col][:]
        for col in self.mis_col:
            X[col+'_miss'] = 0
            X[col+'_miss'][list(X[np.isnan(X[col])].index)] = 1
        xt = DataFrameImputer().fit_transform(Mis)
        for mis in self.mis_col:
            X[mis]=xt[mis]
        
        if len(self.train_cols) == 0:
            self.train_cols = X.columns
        rem = set(self.train_cols) - set(X.columns)
        # below runs for test set only
        for col in rem: 
            X[col] = 0
        X = X[self.train_cols] # keep order same for test and train
        return(X)

    def prepTrainData(self,X,y):
        """Prepare Train and Test data
        Args:
         prepData: datframe with input data
        Output:
         X_train: independent features set of the train set
         X_test: independent features set of the test set
         y_train: Output labels for the train set
         y_test: Output labels for the test set
        """
        
        X_train, X_test, y_train, y_test = train_test_split( X , y , test_size=0.2, random_state=42)
        X_train, y_train = self.handleImbalance(X_train, y_train)
        return(X_train, X_test, y_train, y_test)

    


