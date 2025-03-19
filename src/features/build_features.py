import os
import logging
import click
import pandas as pd

from src.utils.features_manager import get_features_by_property, update_feature_params
from src.utils.transformers import BuildFeatures

@click.command()
@click.option('--datasets', multiple=True, default=['train', 'test'], help='Name of the datasets to be processed')
@click.option('--features_config_path', default='src/data/config/features.yaml', help='Path to the features configuration file')
def main(datasets, features_config_path):
    """
    Process the specified datasets to build features.

    Args:
        datasets (tuple): Names of the datasets to be processed.
    """
    logger = logging.getLogger('Build-Features')
    logger.info('Starting the feature building process')
    
    click.echo(f'Datasets to be processed: {datasets}')
    click.echo(f'Features configuration path: {features_config_path}')
    
    for dataset_name in datasets:
        df = pd.read_parquet(f'data/interim/{dataset_name}.parquet')
        logger.info(f'Dataset {dataset_name} loaded. Shape: {df.shape}')
        raw_features = df.columns.tolist()

        logger.info('Applying the BuildFeatures transformer')
        ratio_features = [
            feat for feat in get_features_by_property(features_config_path, property_name="role", property_value="descriptive") # descriptive features
            if feat in get_features_by_property(features_config_path, property_name="type", property_value="numerical") # numerical features
            and feat != "monto" # exclude target feature
            and feat not in get_features_by_property(features_config_path, property_name="created", property_value=True) # exclude already created features 
        ]
        transformer = BuildFeatures(inference=False, ratio_features=ratio_features)
        df = transformer.fit_transform(df)
        new_features = [col for col in df.columns if col not in raw_features]

        logger.info(f'Saving the dataset {dataset_name} with new features. Shape: {df.shape}')
        path_output = f'data/processed/{dataset_name}.parquet'
        if os.path.exists(path_output):
            os.remove(path_output)
        df.to_parquet(path_output, index=False)
        
    logger.info('Updating the features configuration file')
    for feature in new_features:
        update_feature_params(
            yaml_path='src/data/config/features.yaml',
            feature_name=feature,
            new_params={"type": "numerical", "role": "descriptive", "created": True, "fill_numeric_missing": True}
        )
        
    logger.info('Processing successfully finished!')    
    
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main()