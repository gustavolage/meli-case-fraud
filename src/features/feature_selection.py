import logging
import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from boruta import BorutaPy
import warnings

from src.utils.features_manager import get_features_by_property, update_feature_params
from src.utils.transformers import OptBinningEncoder, NumericMissing

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

def save_feature_importance_plot(model):
    import matplotlib.pyplot as plt
    importances = pd.Series(model.feature_importances_, index=model.feature_names_in_).sort_values(ascending=False)
    fig = importances.plot(kind="bar", figsize=(10, 5))
    fig.set_title("Feature Importance")
    fig.set_ylabel("Importance")
    fig.set_xlabel("Feature")
    plt.savefig("reports/figures/feature_importance.png")
    

@click.command()
@click.option('--dataset', default='train', help='Name of the dataset to be used')
@click.option('--features_config_path', default='src/data/config/features.yaml', help='Path to the features configuration file')
def main(dataset, features_config_path):
    """
    Realize the feature selection process.
    
    Args:
        dataset (str): Name of the dataset to be used.
        features_config_path (str): Path to the features configuration file.    
    """
    logger = logging.getLogger('Feature-Selection')
    logger.info('Starting the feature selection process')
    
    click.echo(f'Dataset to be used: {dataset}')
    click.echo(f'Features configuration path: {features_config_path}')
    
    df = pd.read_parquet(f'data/processed/{dataset}.parquet')
    logger.info(f'Dataset {dataset} loaded. Shape: {df.shape}')
    
    features_to_select = get_features_by_property(yaml_path=features_config_path, property_name="role", property_value="descriptive")
    features_to_select = [
        feature for feature in features_to_select 
        if feature not in get_features_by_property(yaml_path=features_config_path, property_name="hard_remove")
    ]
    logger.info(f'Features to be selected: {features_to_select}')
    
    X = df[features_to_select]
    y = df["fraude"]
    
    features_to_encode = get_features_by_property(yaml_path=features_config_path, property_name="encode", property_value=True)
    features_to_encode = [
        feature for feature in features_to_encode 
        if feature not in get_features_by_property(yaml_path=features_config_path, property_name="hard_remove")
    ]
    logger.info(f'Features to be encoded: {features_to_encode}')
    encoder = OptBinningEncoder(features=features_to_encode)
    encoder.fit(X, y)
    X = encoder.transform(X)
        
    features_to_fill_missing = get_features_by_property(yaml_path=features_config_path, property_name="fill_numeric_missing", property_value=True)
    features_to_fill_missing = [
        feature for feature in features_to_fill_missing
        if feature not in get_features_by_property(yaml_path=features_config_path, property_name="hard_remove")
    ]
    logger.info(f'Features to fill missing values: {features_to_fill_missing}')
    numeric_missing_encoder = NumericMissing(features_to_fill_missing)
    X = numeric_missing_encoder.fit_transform(X)
    
    # Defining the Random Forest model
    random_forest = RandomForestClassifier(
        n_jobs=-1, 
        class_weight="balanced", 
        max_depth=3, 
        random_state=911
    )

    logger.info('Starting the Boruta feature selection process')
    boruta_selector = BorutaPy(
        estimator=random_forest,
        n_estimators=100,
        alpha=0.01,
        max_iter=100,
        random_state=911,
        verbose=2
    ).fit(X.values, y.values)
    
    boruta_selection = X.columns[boruta_selector.support_].tolist()
    logger.info(f'# {len(boruta_selection)} Features selected by Boruta: {boruta_selection}')
    for feature in boruta_selection:
        update_feature_params(
            yaml_path=features_config_path,
            feature_name=feature,
            new_params={"selected_by_boruta": True}
        )
    
    logger.info('Starting the RFECV feature selection process')
    rfecv_selector = RFECV(
        estimator=random_forest, 
        step=1, 
        cv=3, 
        scoring="roc_auc", 
        n_jobs=1,
        verbose=2
    )

    rfecv_selector.fit(X[boruta_selection].values, y.values)
    rfecv_selection = X[boruta_selection].columns[rfecv_selector.support_].tolist()
    logger.info(f'# {len(rfecv_selection)} Features selected by RFECV: {rfecv_selection}')
    for feature in rfecv_selection:
        update_feature_params(
            yaml_path=features_config_path,
            feature_name=feature,
            new_params={"selected_by_rfecv": True}
        )
    
    logger.info('Saving the Feature Importance Report')
    random_forest.fit(X[rfecv_selection], y)
    save_feature_importance_plot(random_forest)

    logger.info('Processing successfully finished!')    
    
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main()