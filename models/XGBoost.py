from models.GridEval import GridEval
from xgboost import XGBClassifier
import numpy as np

class XGBoost(GridEval):
    """run the XGBoost model
    """

    def xgbRunGrid(self):
        """Set the parameters to select from for Random Grid Search 
        Output:
        the best model returned by random grid search
        """
        print("\n"+"XGBOOST CLASSIFIER"+"\n")
        learning_rate = [ i for i in np.linspace(0.0001,0.1,5)]
        n_estimators = [i for i in range(10,100,20)]
        eval_metric = ['logloss']
        seed = [25]
        max_depth = [x for x in range(1,30,10)]
        gamma = [ i for i in np.linspace(1.0,10.0,5)]
        lambdal = [ i for i in np.linspace(1.0,10.0,5)]
        alpha = [ i for i in np.linspace(1.0,10.0,5)]
        feature_selector = ['best']
        objective = ['binary:logistic']
        n_jobs = [-1]
        nthreads = [-1]
        rate_drop = [ i for i in np.linspace(0.4,0.9,3)]
        xgb_param = {
            'learning_rate':learning_rate,
            'n_estimators':n_estimators,
            'eval_metric':eval_metric,
            'seed':seed,
            'max_depth':max_depth,
            'gamma':gamma,
            'objective':objective,
            'nthreads':nthreads,
            'n_jobs':n_jobs,
            'lambda':lambdal,
            'alpha':alpha,
            'feature_selector':feature_selector,
            'rate_drop':rate_drop
        }
        xgb =  XGBClassifier()
        return(self.randomGrid(xgb,xgb_param,5))


    def xgbBest(self):
        """define the best preforming xgboost model for application_train.csv
        """
        print("\n"+"XTREME GRADIENT BOOSTING"+"\n")
        self.model = XGBClassifier(alpha=7.75, base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, eval_metric='logloss',
              feature_selector='best', gamma=10.0,
              learning_rate=0.1, max_delta_step=0, max_depth=11,
              min_child_weight=1, missing=None, n_estimators=30, n_jobs=-1,
              nthread=None, nthreads=-1, objective='binary:logistic',
              random_state=0, rate_drop=0.9, reg_alpha=0, reg_lambda=1,
              scale_pos_weight=1, seed=25, silent=None, subsample=1,
              verbosity=1)
        self.evaluate()