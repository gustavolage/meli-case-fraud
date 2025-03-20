import logging
import click
import pandas as pd
import warnings
import pickle
import os
from sklearn.pipeline import Pipeline

from src.utils.transformers import BuildFeatures, JsonToDataFrame

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

def load_encoder(encoder):
    if encoder == "model":
        with open(os.path.join("models", "predictors", f"{encoder}.pkl"), "rb") as f:
            encoder = pickle.load(f)
    else:
        with open(os.path.join("models", "encoders", f"{encoder}.pkl"), "rb") as f:
            encoder = pickle.load(f)
    return encoder

@click.command()
def main():
    logger = logging.getLogger('Generate-Artifacts')
    logger.info('Starting the generation of artifacts')
    
    logger.info('Loading encoders')
    feature_type = load_encoder("feature_type")
    fill_numeric = load_encoder("fill_numeric")
    string_encoder = load_encoder("string_encoder")
    selector = load_encoder("selector")
    
    
    logger.info('Loading the model')
    model = load_encoder("model")
    
    # Updadating build_features 
    build_features = BuildFeatures(inference=True, ratio_features = ['a', 'b', 'c', 'd', 'e', 'f', 'h', 'k', 'l', 'm', 'hour', 'weekday'])
    
    # Creating json to dataframe transformer
    json_to_df = JsonToDataFrame()
    
    # Prdouction pipeline
    model_pipeline_prod = Pipeline([
        ('json_to_df', json_to_df),
        ('features_type', feature_type),
        ('build_features', build_features),
        ('fill_numeric', fill_numeric),
        ('string_encoder', string_encoder),
        ('selector', selector),
        ('model', model)
    ])
    
    # Testing pipeline
    model_pipeline_test = Pipeline([
        ('features_type', feature_type),
        ('build_features', build_features),
        ('fill_numeric', fill_numeric),
        ('string_encoder', string_encoder),
        ('selector', selector),
        ('model', model)
    ])    
    
    logger.info('Saving the final model pipelines')
    for name, pipeline in zip(["prod",  "test"], [model_pipeline_prod, model_pipeline_test]):
        with open(os.path.join("models", "wrapped", f"model_pipeline_{name}.pkl"), "wb") as f:
            pickle.dump(pipeline, f)
    
    logger.info('Artifacts generated successfully')    
    
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main()