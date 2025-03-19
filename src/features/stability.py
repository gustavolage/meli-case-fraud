import logging
import click
import pandas as pd
import warnings

from src.utils.features_manager import get_features_by_property, update_feature_params
from src.utils.stability_report import StabilityReport

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

@click.command()
@click.option('--dataset', default='train', help='Name of the dataset to be used')
@click.option('--features_config_path', default='src/data/config/features.yaml', help='Path to the features configuration file')
@click.option('--selector', default='boruta', help='The selector used')
def main(dataset, features_config_path, selector):
    """
    Realize the generation of the stability report.
    
    Args:
        dataset (str): Name of the dataset to be used.
        features_config_path (str): Path to the features configuration file.    
        selector (str): The selector used.    
        
    """
    logger = logging.getLogger('Stability-Report')
    logger.info('Starting the generation of the stability report')
    
    click.echo(f'Dataset to be used: {dataset}')
    click.echo(f'Features configuration path: {features_config_path}')
    click.echo(f'Selector: {selector}')
    
    df = pd.read_parquet(f'data/processed/{dataset}.parquet')
    logger.info(f'Dataset {dataset} loaded. Shape: {df.shape}')
    
    features_selected = get_features_by_property(features_config_path, f"selected_by_{selector}")
    logger.info(f'# {len(features_selected)} Features: {features_selected}')

    logger.info('Generating Stability Report - Features Selected')
    report = StabilityReport(df)
    report.generate_pdf_report(
        col_tempo="week_of_the_year",
        features_config_path=features_config_path,
        variables=features_selected + ["fraude"],
        output_path="reports/stability_report.pdf"
    )
    
    logger.info('Processing successfully finished!')    
    
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main()