import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import time
t0 = time.time()
from sklearn.metrics import *
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


class GridEval():
    """run the Grid Search on the train set and show the evaluation results
    """
    def __init__(self,xtrain,xtest,ytrain,ytest):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.model = None
        self.ypred_test = None

    def evaluate(self):
        """fit the model and show the evaluation results for the train and test set.
        """
        self.model.fit(self.xtrain,self.ytrain)
        print("Final model has been fit")
        ypred_test = self.model.predict(self.xtest)
        ypred_train = self.model.predict(self.xtrain)
        metric=pd.DataFrame([[0]*4]*2,columns=['PRECISION','RECALL','F1_SCORE','ACCURACY'],index=['Train','Test'],dtype='float32')
        metric['PRECISION']['Train']=precision_score(self.ytrain, ypred_train)
        metric['RECALL']['Train']=recall_score(self.ytrain, ypred_train)
        metric['F1_SCORE']['Train']=f1_score(self.ytrain, ypred_train)
        metric['ACCURACY']['Train']=accuracy_score(self.ytrain,ypred_train)
        metric['PRECISION']['Test']=precision_score(self.ytest, ypred_test)
        metric['RECALL']['Test']=recall_score(self.ytest, ypred_test)
        metric['F1_SCORE']['Test']=f1_score(self.ytest, ypred_test)
        metric['ACCURACY']['Test']=accuracy_score(self.ytest,ypred_test)
        print(metric)
        print("Train confusion matrix")
        print(confusion_matrix(self.ytrain,ypred_train))
        print("Test confusion matrix")
        print(confusion_matrix(self.ytest,ypred_test))
        print("Time taken to fit: "+str(time.time()-t0))


    def randomGrid(self,typeModel,param,niter):
      """run random search
      Args:
       typeModel: One of the 6 models - KNN, Dec Tree, RandForest, Logistic, XGBoost, LightGBM
      Output:
       retmodel: Best mdoel found by random search CV
      """
      print("Finding best parameters for the model")
      print("train shape",self.xtrain.shape,"test shape",self.xtest.shape)
      self.model =  RandomizedSearchCV(estimator = typeModel, param_distributions=param, n_iter = niter, cv=3, scoring='f1',random_state=42,n_jobs = -1,verbose =2)    
      self.evaluate() 
      print("\nBest parameters: ")
      print(self.model.best_estimator_)
      retmodel = self.model
      return(retmodel)

    def writeResults(self,filename,inputX):
        """write output labels to filename Output.csv
        Args:
         filename: target filename
         inputX: test data input
        """
        yPred = self.model.predict(inputX)
        with open(filename, "w") as outfile:
          outfile.write("TARGET"+"\n")
          for y in yPred:
            outfile.write(str(y)+"\n")
        print("Output written to {}".format(filename))

