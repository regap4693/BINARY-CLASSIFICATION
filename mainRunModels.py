#!/usr/bin/python3
from process_data.Preprocess import Preprocess
from models.DecisionTree import DecisionTree
from models.KNearest import KNearest
from models.Logistic import Logistic
from models.RandomForest import RandomForest
from models.LightGBM import LightGBM
from models.XGBoost import XGBoost
import argparse
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

"""
Description:
This program tries to run different models with different techniques to balance data - both based on user choice
The different choices available can be seen in the choices and help section in the parser argument details below
It prints the resulting evaluation metrics - Precision, Recall, F1score and Accuracy
It also writes the predicted values (0 or 1) for any test file (which should be same format as the train file) into Output.csv
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--isNewData', default='no', type=str, choices = ['yes','no'], help="If new data then need to run the grid to determine best parameters")
    parser.add_argument('--balance', default=None, type=str, choices = ["over","under","smote","classWeights"], help="Choose Imbalance strategy. Note classWeights is not applicable for KNN. classWeights requires class_wt arguement to be passed")
    parser.add_argument('--modeltype', default='Logistic',type=str, choices=["KNN","DecisionTree","RandomForest","Logistic","XGBoost","LightGBM"], help= "Choose which model to run. Defaults to Logistic")
    parser.add_argument('--test_file', default=None, type=str, help="Test file (path or name) to generate output labels in Output.csv. By default no output is written.")
    args = parser.parse_args()

    """initialize variables and class objects
    """
    srcPath = "application_train.csv"
    testPath = args.test_file

    if args.isNewData == 'yes':
        grid = True
    else:
        grid = False

    modeltype = args.modeltype

    """set the class weights only if the user has chosen classWeights
    """
    if args.balance == "classWeights":
        class_wt = 'balanced'
        pre = Preprocess()
    else:
        class_wt = None
        pre = Preprocess(args.balance)


    """Preprocess the data
    """
    data = pd.read_csv(srcPath)
    y = data['TARGET']
    X = data.drop('TARGET',axis=1)
    X = pre.readPreprocess(X).values

    X_train, X_test, y_train, y_test = pre.prepTrainData(X,y)

    """run the models based on the user input
    """
    if modeltype == 'KNN':
        obj = KNearest(X_train, X_test, y_train, y_test)
        if grid:
            model = obj.knnRunGrid()
        else:
            model = obj.knnBest()
    elif modeltype == 'DecisionTree':
        obj = DecisionTree(X_train, X_test, y_train, y_test)
        if grid:
            model = obj.decisionRunGrid(class_wt)
        else:
            model = obj.decisionBest()
    elif modeltype == 'RandomForest':
        obj = RandomForest(X_train, X_test, y_train, y_test)
        if grid:
            model = obj.randomRunGrid(class_wt)
        else:
            model = obj.randomBest()
    elif modeltype == 'Logistic':
        obj = Logistic(X_train, X_test, y_train, y_test)
        if grid:
            model = obj.logisticRunGrid(class_wt)
        else:
            model = obj.logisticBest()
    elif modeltype == 'LightGBM':
        obj = LightGBM(X_train, X_test, y_train, y_test)
        if grid:
            model = obj.lgbRunGrid()
        else:
            model = obj.lgbBest()
    else:
        obj = XGBoost(X_train, X_test, y_train, y_test)
        if grid:
            model = obj.xgbRunGrid()
        else:
            model = obj.xgbBest()

    """output the results for the test file if any
    """
    if testPath:
        inpData = pd.read_csv(testPath)
        data = pre.readPreprocess(inpData).values
        obj.writeResults("Output.csv", data)
        

if __name__ == "__main__":
    main()