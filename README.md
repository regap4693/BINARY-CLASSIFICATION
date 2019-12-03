# BINARY-CLASSIFICATION
- This program tries to run different models with different techniques to balance data - both based on user choice
- The different choices available can be seen in the choices and help section in the parser argument details below
- It prints the resulting evaluation metrics - Precision, Recall, F1score and Accuracy
- It also writes the predicted values (0 or 1) for any test file (which should be same format as the train file) into Output.csv

## Install required packages:
1. Install latest version of python3.
2. create a virtual environment:
   - virtualenv env_name
3. Activate the virtual environment:
   - source env_name/bin/activate
4. Check python version- output should be Python 3.7.3:
   - python --version
4. install the python packages:
   - pip install -r requirements.txt

## Dataset
Below has the Train dataset "application_train.csv" required to run the program.
https://drive.google.com/drive/folders/1ZKjQa4dTV-Eee2O_chH0X_caHykYeKCM?usp=sharing
Optionally you can download the "application_test.csv" to obtain the output labels.

## How to run:

- Example command line:
For Undersampling:
- python mainRunModels.py --balance under --modeltype Logistic
- python mainRunModels.py --balance under --modeltype KNN
- python mainRunModels.py --balance under --modeltype DecisionTree
- python mainRunModels.py --balance under --modeltype RandomForest
- python mainRunModels.py --balance under --modeltype Xgboost
- python mainRunModels.py --balance under --modeltype LightGBM

Similarly for Oversampling, SMOTE
- python mainRunModels.py --balance over --modeltype Logistic
- python mainRunModels.py --balance smote --modeltype KNN

Not passing any parameter trains the model without balancing the train dataset. Hence, below gives very low f1score when compared to when it is balanced.
- python mainRunModels.py --modeltype Logistic

- We apply three imbalance techniques (--balance) and 6 models that were experimented with (--modeltype). 

Options availabe for modeltype:
KNN,DecisionTree,RandomForest,Logistic,Xgboost,LightGBM

Below are the options available for the argument --balance:

over:

Oversampling - In this method the data from the minority class is replicated so that a balance can be established between the two classes.

under:

Undersampling - In this method the data from the majority class is removed in order to balance the data.

smote:

Synthetic minority oversampling technique - The over sampling of the minority dataset is done synthetically using an algorithm. Data from the minority class is not replicated. This prevent over fitting which is prevalent in oversampling.

classWeights:

Modify the cost function - There are several approaches-[1] one which we are using is class weights. The class weights are added to the minority class so that the cost function accounts more for error in minority class.
