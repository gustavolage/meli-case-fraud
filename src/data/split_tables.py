import os
import logging
import click
import pandas as pd

@click.command()
@click.option('--first_week_test_set', type=int, default=16, help='First week of the year that will be used as test set')
def main(first_week_test_set):
    logger = logging.getLogger('Split-Tables')
    logger.info('Starting the basic process')
    
    df = pd.read_parquet('data/interim/dados.parquet')
    logger.info(f'Dataset loaded. Shape: {df.shape}')
    
    click.echo(f'Week of the year that starts test set: {first_week_test_set}')

    df_train = df[df['week_of_the_year'] < first_week_test_set]
    logger.info(f'Saving the train interim dataset. Shape: {df_train.shape} ({df_train.shape[0] / df.shape[0]:.2%})')
    logger.info(f'# {df_train.fraude.sum()} frauds')
    path_output = 'data/interim/train.parquet'
    if os.path.exists(path_output):
        os.remove(path_output)
    df_train.to_parquet(path_output, index=False)
    
    df_test = df[df['week_of_the_year'] >= first_week_test_set]
    logger.info(f'Saving the test interim dataset. Shape: {df_test.shape} ({df_test.shape[0] / df.shape[0]:.2%})')
    logger.info(f'# {df_test.fraude.sum()} frauds')
    path_output = 'data/interim/test.parquet'
    if os.path.exists(path_output):
        os.remove(path_output)
    df_test.to_parquet(path_output, index=False)
    
    logger.info('Processing successfully finished!')        
    
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main()