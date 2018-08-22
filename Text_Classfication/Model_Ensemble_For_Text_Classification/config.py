import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--max_len', type=int, default=300)
parser.add_argument('--mode', type=str, default="test")

args = parser.parse_args()



rf_params = {
    'n_jobs': 5,
    'n_estimators': 200,
     'warm_start': False,
     #'max_features': 0.2,
    'max_depth': 30,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_estimators':200,
    #'max_features': 0.5,
    'max_depth': 10,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 200,
    'learning_rate' : 0.75,
}

# Gradient Boosting parameters
gb_params = {
    'max_features' : 'sqrt',
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_leaf': 2,
    'verbose': 0,
    #'max_features ':0.4
}

# Support Vector Classifier parameters
svc_params = {
    'kernel' : 'rbf',
    'C' : 0.8
    }

lg_params={
    "max_iter":100,
    "C":1.0,
    "penalty":'l2'
}