import logging
import click
import pandas as pd
import warnings
import os
import pickle

from src.utils.features_manager import get_features_by_property, update_feature_params
from src.utils.transformers import (
    FeaturesType,
    NumericMissing,
    OptBinningEncoder,
    BuildFeatures,
    Selector
)

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

def apply_encoders_transformations(X, transformers, features_auxiliary):
    """Apply the transformations of the encoders to the dataset.
    """
    df_aux = X[features_auxiliary].copy()
    
    for _, transformer in transformers.items():
        X = transformer.transform(X)

    X[features_auxiliary] = df_aux
    
    return X

@click.command()
@click.option('--datasets', multiple=True, default=['train', 'test'], help='Name of the datasets to be processed')
@click.option('--features_config_path', default='src/data/config/features.yaml', help='Path to the features configuration file')
@click.option('--selector', default='boruta', help='The selector used')
def main(datasets, features_config_path, selector):
    """
    Creating encoders
    
    Args:
        dataset (str): Name of the dataset to be used.
        features_config_path (str): Path to the features configuration file.    
        selector (str): The selector used.    
        
    """
    logger = logging.getLogger('Creating-Encoders')
    logger.info('Starting the generation of encoders')
    
    click.echo(f'Datasets to be used: {datasets}')
    click.echo(f'Features configuration path: {features_config_path}')
    click.echo(f'Selector: {selector}')
    
    df = pd.read_parquet(f'data/interim/{datasets[0]}.parquet')
    logger.info(f'Dataset {datasets[0]} loaded. Shape: {df.shape}')
    
    logger.info('Features Type')
    features_dtypes = {
        "binary": get_features_by_property(features_config_path, property_name="type", property_value="binary"),
        "categorical": get_features_by_property(features_config_path, property_name="type", property_value="categorical"),
        "numerical": get_features_by_property(features_config_path, property_name="type", property_value="numerical"),
        "datetime": get_features_by_property(features_config_path, property_name="type", property_value="datetime")
    }
    
    for dtype in features_dtypes:
        features_dtypes[dtype] = [
            feature for feature in features_dtypes[dtype]
            if feature in get_features_by_property(features_config_path, "created", False)
            and feature in df.columns
        ]
        
    features_type = FeaturesType(dtypes=features_dtypes)
    features_type.fit(df)
    df = features_type.transform(df)
    logger.info('Features Type successfully created!')
    
    
    logger.info('Build Features')
    ratio_features = [
        feat for feat in get_features_by_property(features_config_path, property_name="role", property_value="descriptive") # descriptive features
        if feat in get_features_by_property(features_config_path, property_name="type", property_value="numerical") # numerical features
        and feat != "monto" # exclude target feature
        and feat in df.columns # check if feature is in the dataframe 
    ]
    build_features = BuildFeatures(inference=False, ratio_features=ratio_features)
    build_features.fit(df)
    df = build_features.transform(df)
    logger.info('Build Features successfully created!')
    
    
    logger.info('Fill Numeric Missing')
    features_fill_numeric = get_features_by_property(features_config_path, "fill_numeric_missing", True)
    fill_numeric = NumericMissing(features_fill_numeric)
    fill_numeric.fit(df)
    df = fill_numeric.transform(df)
    logger.info('Fill Numeric Missing successfully created!')
    
    logger.info('OptBinningEncoder')
    features_to_encode = get_features_by_property(yaml_path=features_config_path, property_name="encode", property_value=True)
    string_encoder = OptBinningEncoder(features_to_encode)
    string_encoder.fit(df, df["fraude"])
    df = string_encoder.transform(df)
    logger.info('OptBinningEncoder successfully created!')
    
    logger.info('Selector')
    features_selected = get_features_by_property(yaml_path=features_config_path, property_name=f"selected_by_{selector}", property_value=True)
    selector = Selector(features=features_selected)
    selector.fit(df)
    df = selector.transform(df)
    logger.info('Selector successfully created!')
    
    logger.info('Applying transformations')
    transformers = {
        "feature_type": features_type,
        "build_features": build_features,
        "fill_numeric": fill_numeric,
        "string_encoder": string_encoder,
        "selector": selector
    }
    features_auxiliary = get_features_by_property(yaml_path=features_config_path, property_name="role", property_value="auxiliary")
    features_auxiliary = [feat for feat in features_auxiliary if feat in df.columns]
    for dataset in datasets:
        logger.info(f'Processing dataset {dataset}')
        df = pd.read_parquet(f'data/interim/{dataset}.parquet')
        df_transformed = apply_encoders_transformations(df, transformers, features_auxiliary)
        logger.info(f'Saving the dataset {dataset} with transformations. Shape: {df_transformed.shape}')
        path_output = f'data/processed/encoded_{dataset}.parquet'
        if os.path.exists(path_output):
            os.remove(path_output)
        df_transformed.to_parquet(path_output, index=False)
    logger.info('Transformations successfully applied!')
    
    logger.info('Saving the binary encoders')
    for encoder, transformer in transformers.items():
        pickle.dump(transformer, open(f'models/encoders/{encoder}.pkl', 'wb'))
    logger.info('Binary encoders successfully saved!')
        
    
    logger.info('Processing successfully finished!')    
    
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main()