import yaml

def get_features_by_property(yaml_path, property_name, property_value=None):
    with open(yaml_path, 'r') as f:
        features = yaml.safe_load(f)
    
    result = []
    for feature_name, attrs in features.items():
        if property_name in attrs:
            if property_value is not None:
                if attrs[property_name] == property_value:
                    result.append(feature_name)
            else:
                result.append(feature_name)
    return result


def update_feature_params(yaml_path, feature_name, new_params, output_path=None):
    with open(yaml_path, 'r') as f:
        features = yaml.safe_load(f)
    
    if feature_name not in features:
        features[feature_name] = {}
    
    features[feature_name].update(new_params)
    
    target_path = output_path if output_path is not None else yaml_path
    
    with open(target_path, 'w') as f:
        yaml.dump(features, f, default_flow_style=False, sort_keys=False)
    
    return features