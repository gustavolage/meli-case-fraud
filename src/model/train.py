import logging
import click
import pandas as pd
import warnings
import optuna
import numpy as np
import yaml
import pickle
import os
import lightgbm as lgb

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from src.utils.features_manager import get_features_by_property
from src.utils.optuna_settings import objective

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

def load_encoder(encoder):
    with open(os.path.join("models", "encoders", f"{encoder}.pkl"), "rb") as f:
        encoder = pickle.load(f)
    return encoder

def partial_roc_auc(y_true, y_score, max_fpr=0.05):
    return roc_auc_score(y_true, y_score, max_fpr=max_fpr)

def apply_transformers(df, transformers):
    for transformer in transformers:
        df = transformer.transform(df)
    return df

@click.command()
@click.option('--dataset', default='train', help='Name of the dataset to be used')
@click.option('--features_config_path', default='src/data/config/features.yaml', help='Path to the features configuration file')
@click.option('--selector', default='rfecv', help='The selector used')
def main(dataset, features_config_path, selector):
    """
    Model training using the best hyperparameters found by Optuna.
    """
    logger = logging.getLogger('Train-Model')
    logger.info('Starting model training')
    
    df_train = pd.read_parquet(f'data/processed/{dataset}.parquet')
    logger.info(f'Dataset {dataset} loaded with shape: {df_train.shape}')
    
    target_col = get_features_by_property(features_config_path, 'target')[0]
    
    logger.info('Loading encoders')
    fill_numeric = load_encoder("fill_numeric")
    string_encoder = load_encoder("string_encoder")
    selector = load_encoder("selector")
    
    logger.info('Splitting the dataset into train and validation sets')
    df_train, df_val = train_test_split(df_train, test_size=0.15, random_state=911, stratify=df_train[target_col])
    print(f"Shape train: {df_train.shape} | # {df_train['fraude'].sum()} fraudes | Bad rate: {df_train['fraude'].mean():.2%}")
    print(f"Shape val: {df_val.shape} | # {df_val['fraude'].sum()} fraudes | Bad rate: {df_val['fraude'].mean():.2%}")
    
    # Apply transformers
    transformers = [fill_numeric, string_encoder]
    # Refit string encoder to avoid data leakage
    string_encoder.fit(df_train, df_train[target_col])
    
    df_train = apply_transformers(df_train, transformers)
    df_val = apply_transformers(df_val, transformers)
    
    logger.info('Training the final model with the best hyperparameters')
    best_params = yaml.safe_load(open('src/model/optuna_tuning_results/best_params.yaml', 'r'))["params"]

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        class_weight='balanced',
        early_stopping_rounds=10, # Small number to avoid overfitting
        **best_params,
        verbose=1
    )
    model.fit(
        df_train[selector.features],
        df_train[target_col],
        eval_set=[(df_val[selector.features], df_val[target_col])],
        eval_metric='auc',
    )
    
    logger.info('Evaluating the final model in Train set')
    y_train_pred = model.predict_proba(df_train[selector.features])[:, 1]
    roc_auc_train = roc_auc_score(df_train[target_col], y_train_pred) * 100
    roc_auc_train_partial = partial_roc_auc(df_train[target_col], y_train_pred) * 100
    logger.info(f"ROC_AUC: {roc_auc_train:.2f}% | ROC_AUC@0.05: {roc_auc_train_partial:.2f}%")
    
    logger.info('Evaluating the final model in Validation set')
    y_vali_pred = model.predict_proba(df_val[selector.features])[:, 1]
    roc_auc_val = roc_auc_score(df_val[target_col], y_vali_pred) * 100
    roc_auc_val_partial = partial_roc_auc(df_val[target_col], y_vali_pred) * 100
    logger.info(f"ROC_AUC: {roc_auc_val:.2f}% | ROC_AUC@0.05: {roc_auc_val_partial:.2f}%")
    
    logger.info('Saving the final model')
    with open('models/predictors/model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    logger.info('Model trained saved successfully!')
    
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main()
