import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def objective(trial, X_train, y_train, X_val, y_val, max_fpr):
    param = {
        'n_estimators': 1000,  # Large number for early stopping
        'class_weight': 'balanced',
        'early_stopping_rounds': 10,
        'verbosity': -1,
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'num_leaves': trial.suggest_int('num_leaves', 2, 60)
    }

    model = lgb.LGBMClassifier(**param)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc'
    )

    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_val_pred = model.predict_proba(X_val)[:, 1]

    roc_auc_partial_train = roc_auc_score(y_train, y_train_pred, max_fpr=max_fpr)
    roc_auc_partial_val = roc_auc_score(y_val, y_val_pred, max_fpr=max_fpr)

    metric_difference = abs(roc_auc_partial_train - roc_auc_partial_val)
        
    return roc_auc_partial_val, metric_difference
