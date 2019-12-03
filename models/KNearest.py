from models.GridEval import GridEval
from sklearn.neighbors import KNeighborsClassifier

class KNearest(GridEval):
    """run the KNearest Neighbors model
    """

    def knnRunGrid(self):
        """Set the parameters to select from for Random Grid Search 
        Output:
        the best model returned by random grid search
        """
        print("\n"+"KNEAREST CLASSIFIER"+"\n")
        knn_param = {'algorithm':['auto'],              
                  "n_neighbors" : [i for i in range(1, 10)],
                   "n_jobs" : [-1] } 
        knn =  KNeighborsClassifier()
        return(self.randomGrid(knn,knn_param,7))

    def knnBest(self):
        """define the best performing KNearest Neighbor model for application_train.csv
        """
        print("\n"+"KNEAREST CLASSIFIER"+"\n")
        self.model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=-1, n_neighbors=9, p=2,
                     weights='uniform')
        self.evaluate()
