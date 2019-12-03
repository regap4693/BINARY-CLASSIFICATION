from models.GridEval import GridEval
from lightgbm import LGBMClassifier
import numpy as np
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

class LightGBM(GridEval):
    """run the LightGBM model
    """

    def lgbRunGrid(self):
        """Set the parameters to select from for Random Grid Search 
        Output:
        the best model returned by random grid search
        """
        print("\n"+"LIGHT GRADIENT BOOSTING"+"\n")
        learning_rate = [ i for i in np.linspace(0.0001,0.1,5)]
        n_estimators = [i for i in range(10,200,40)]
        metric = ['binary']
        seed = [25]
        max_depth = [x for x in range(1,30,10)]
        alpha = [ i for i in np.linspace(1.0,10.0,5)]
        objective = ['binary']
        n_jobs = [-1]
        nthreads = [-1]
        nthread = [-1]
        min_data_in_leaf = [x for x in range(1,30,6)]
        rate_drop = [ i for i in np.linspace(0.4,0.9,3)]
        lambdal = [ i for i in np.linspace(1.0,10.0,5)]
        num_leaves = [i for i in range(100,1000,200)]
        lgb_param = {
            'learning_rate':learning_rate,
            'n_estimators':n_estimators,
            'objective':objective,
            'metric':metric,
            'seed':seed,
            'max_depth':max_depth,
            'nthreads':nthreads,
            'n_jobs':n_jobs,
            'lambda':lambdal,
            'alpha':alpha,
            'rate_drop':rate_drop,
            'min_data_in_leaf':min_data_in_leaf,
            'num_leaves':num_leaves 

        }
        lgb =  LGBMClassifier()
        return(self.randomGrid(lgb,lgb_param,10))

    def lgbBest(self):
        """define the best performing LightGBM model for application_train.csv
        """
        print("\n"+"LIGHT GRADIENT BOOSTING"+"\n")
        self.model = LGBMClassifier(alpha=3.25, boosting_type='gbdt', class_weight=None,
               colsample_bytree=1.0, importance_type='split',
               learning_rate=0.07502500000000001, max_depth=21, metric='binary',
               min_child_samples=20, min_child_weight=0.001, min_data_in_leaf=7,
               min_split_gain=0.0, n_estimators=170, n_jobs=-1, nthreads=-1,
               num_leaves=100, objective='binary', random_state=None,
               rate_drop=0.65, reg_alpha=0.0, reg_lambda=0.0, seed=25,
               silent=True, subsample=1.0, subsample_for_bin=200000,
               subsample_freq=0)
        self.evaluate()