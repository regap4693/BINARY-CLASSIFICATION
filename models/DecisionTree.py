from models.GridEval import GridEval
from sklearn import tree
class DecisionTree(GridEval):
    """run the Decision Tree model
    """

    def decisionRunGrid(self, classWeight = None):
        """Set the parameters to select from for Random Grid Search 
        Output:
        the best model returned by random grid search
        """
        print("\n"+"DECISION TREE"+"\n")
        max_features = ['auto']
        criterion = ['gini']
        max_depth = [x for x in range(1,50,15)]
        dec_param= {    'max_features': max_features,
                        'max_depth': max_depth,
                        'criterion': criterion,
                        'random_state': [42],
                        'class_weight': [classWeight]
                    }
        dectree = tree.DecisionTreeClassifier()
        return(self.randomGrid(dectree,dec_param,5))

    def decisionBest(self):
        """define the best performing decision tree model for application_train.csv
        """
        print("\n"+"DECISION TREE"+"\n")
        self.model = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=46,
                       max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=42, splitter='best')
        self.evaluate()
