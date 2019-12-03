from models.GridEval import GridEval
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

class Logistic(GridEval):
    """run the Logistic Regression model
    """

    def logisticRunGrid(self, classWeight = None):
        """Set the parameters to select from for Random Grid Search 
        Output:
        the best model returned by random grid search
        """
        print("\n"+"LOGISTIC REGRESSION"+"\n")
        penalty = ['l2','l1']
        C_val = [i*0.1 for i in range(2,10,3)]
        log_param = {"C" : C_val,
                   'penalty': penalty,
                    'n_jobs':[-1],
                    'class_weight': [classWeight]
                    }    

        logreg = LogisticRegression()
        return(self.randomGrid(logreg,log_param,2))

    def logisticBest(self):
        """define the best performing logistic regression model for application_train.csv
        """
        print("\n"+"LOGISTIC REGRESSION"+"\n")
        self.model = LogisticRegression(C=0.2, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=-1, penalty='l1',
                   random_state=None, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)
        self.evaluate()