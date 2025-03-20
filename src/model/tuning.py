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
@click.option('--n_trials', default=50, help='Number of trials to be executed')
def main(dataset, features_config_path, selector, n_trials):
    """
    Model tuning using Optuna with multi-objective optimization.
    The metrics used are:
      - roc_auc_partial will be maximized;
      - train-val metric difference will be minimized.

    The hyperparameter configurations should be defined in src/optuna_settings.py.
    """
    logger = logging.getLogger('Optuna-Tuning')
    logger.info('Starting model tuning with Optuna')
    
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
    
    # multi-objective optimization 
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        study_name="model_tuning",
        sampler=optuna.samplers.NSGAIISampler(seed=911), # multi-objective optimization        
    )   
    
    trials = lambda trial: objective(
        trial,
        X_train=df_train[selector.features],
        y_train=df_train[target_col],
        X_val=df_val[selector.features],
        y_val=df_val[target_col],
        max_fpr=0.05
    )
    study.optimize(
        trials,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
        n_jobs=1
    )
    
    best_trial = study.best_trials[0]
    logger.info("Best trial:")
    logger.info(f"Values: {best_trial.values}")
    logger.info(f"Params: {best_trial.params}")
    
    logger.info('Saving best params')
    yaml.dump({'params': best_trial.params}, open('src/model/optuna_tuning_results/best_params.yaml', 'w'))
    logger.info('! Check the trials to see if have another best trial')
    
    logger.info('Saving table with trials results')
    trials_df_train = study.trials_dataframe()
    trials_df_train.to_csv('src/model/optuna_tuning_results/trials.csv', index=False)
    
    logger.info('Optuna tuning successfully finished!')
    
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main()
