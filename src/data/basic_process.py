import os
import logging
import click
import pandas as pd
import datetime

from src.utils.transformers import FeaturesType
from src.utils.features_manager import get_features_by_property, update_feature_params

def get_week_from_date(date: datetime.date) -> int:
    """
    Retorna o número da semana do ano para uma data específica.
    
    Args:
        date (datetime.date): Data para extrair a semana.
    
    Returns:
        int: Número da semana do ano (1-53).
    """
    return date.isocalendar()[1]

@click.command()
@click.option('--features_config_path', default='src/data/config/features.yaml', help='Path to the features configuration file')
def main(features_config_path):
    logger = logging.getLogger('Basic-Process')
    logger.info('Starting the basic process')
    
    click.echo(f'Features configuration path: {features_config_path}')
    
    df = pd.read_csv('data/raw/dados.csv')
    logger.info(f'Dataset loaded. Shape: {df.shape}')
    
    logger.info('Creating index column')
    df['index'] = df.index
    update_feature_params(features_config_path, 'index', {"type": "index", "role": "auxiliary", "created": True})
    logger.info('Saving the raw dataset with the index column')
    path_output = 'data/raw/dados.parquet'
    if os.path.exists(path_output):
        os.remove(path_output)
    df.to_parquet(path_output, index=False)
    
    
    dtypes = {
        "binary": get_features_by_property(features_config_path, property_name="type", property_value="binary"),
        "categorical": get_features_by_property(features_config_path, property_name="type", property_value="categorical"),
        "numerical": get_features_by_property(features_config_path, property_name="type", property_value="numerical"),
        "datetime": get_features_by_property(features_config_path, property_name="type", property_value="datetime")
    }
    
    for dtype in dtypes:
        dtypes[dtype] = [feature for feature in dtypes[dtype] if feature in get_features_by_property(features_config_path, property_name="created", property_value=False)]
    
    logger.info('Applying the FeaturesType transformer')
    transformer = FeaturesType(dtypes)
    df = transformer.fit_transform(df)
    
    logger.info('Removing features that are not going to be used')
    features_to_remove = get_features_by_property(features_config_path, property_name="hard_remove", property_value=True)
    df = df.drop(columns=features_to_remove)
    logger.info(f'Features removed: {features_to_remove}')
    
    logger.info('Removing features with high cardinality (unique more than 1% of the dataset size) - problably ID features')
    n_unique_threshold = int(df.shape[0] * 0.01)
    features_high_cardinality = [feature for feature in df.select_dtypes("object").columns if df[feature].nunique() > n_unique_threshold]
    df = df.drop(columns=features_high_cardinality)
    logger.info(f'Features removed: {features_high_cardinality}')
    
    logger.info('Creating weekend feature')
    df['week_of_the_year'] = pd.to_datetime(df['fecha']).apply(lambda row: get_week_from_date(row))
    update_feature_params(features_config_path, 'week_of_the_year', {"type": "numerical", "role": "auxiliary", "created": True})
    
    logger.info(f'Saving the interim dataset. Shape: {df.shape}')
    path_output = 'data/interim/dados.parquet'
    if os.path.exists(path_output):
        os.remove(path_output)
    df.to_parquet(path_output, index=False)
    
    logger.info('Processing successfully finished!')        
    
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main()