from models.GridEval import GridEval
from sklearn.ensemble import RandomForestClassifier

class RandomForest(GridEval):
    """run the Random Forest model
    """

    def randomRunGrid(self,classWeight = None):
        """Set the parameters to select from for Random Grid Search 
        Output:
        the best model returned by random grid search
        """
        print("\n"+"RANDOM FOREST"+"\n")
        n_estimators = [x for x in range(5,20,4)]
        max_features = ['auto']
        max_depth = [x for x in range(60,95,15)]
        bootstrap = [True]
        rand_param = { 'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'bootstrap': bootstrap,
                       'n_jobs': [-1],
                       'class_weight': [classWeight]
                        }     
        
        randfor = RandomForestClassifier()
        return(self.randomGrid(randfor,rand_param,9))

    def randomBest(self):
        """define the best performing random forest model for application_train.csv
        """
        print("\n"+"RANDOM FOREST"+"\n")
        self.model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=90, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=17, n_jobs=-1,
                       oob_score=False, random_state=None, verbose=0,
                       warm_start=False)
        self.evaluate()