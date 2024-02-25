import json
from striprtf.striprtf import rtf_to_text

with open('algoparams_from_ui.json.rtf', 'r') as file: 
    rtf_text = file.read() 

def convert_rtf_text(rtf_text):
    return rtf_text

plain_text = rtf_to_text(convert_rtf_text(rtf_text))
Input = json.loads(plain_text)


def pretty_print(obj):
    print(json.dumps(obj, indent=2))

print("\nTarget:")
target = Input['design_state_data']['target']
pretty_print(target)

print("\nFeature Handling:")
feature_handling = Input['design_state_data']['feature_handling']
pretty_print(feature_handling)

dataset_name = Input['design_state_data']['session_info']['dataset']
print(f"\nDataset name: {dataset_name}")
import pandas as pd
data = pd.read_csv(dataset_name)


for column, feature in feature_handling.items():
    if not feature['is_selected']:
        data.drop(column, axis=1, inplace=True)

    if feature["feature_variable_type"] == "numerical":

        if feature['feature_details']["missing_values"] == "Impute":
            if feature['feature_details']['impute_with'] == "Average of values":
                data[column].fillna(data[column].mean(), inplace=True)
            elif feature['feature_details']['impute_with'] == "custom":
                data[column].fillna(feature['feature_details']['impute_value'], inplace=True)
            else:
                AssertionError(f"Unknown imputation method: {feature['feature_details']['impute_with']}")
    elif feature["feature_variable_type"] == "text":
        labels = {key: num for num, key in enumerate(data[column].unique())}
        data[column] = data[column].apply(lambda x: labels[x])

    else:
        AssertionError(f"Unknown feature type: {feature['feature_variable_type']}")

# Feature Selection
Feature_selection = Input['design_state_data']['feature_reduction']

target_column = target['target']

X = data.drop(target_column, axis=1).values
y = data[target_column].values

if Feature_selection['feature_reduction_method'] == "Tree-based":
    if target['type'] == "regression":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import SelectFromModel
        sel = SelectFromModel(RandomForestRegressor(n_estimators=int(Feature_selection['num_of_trees']), max_depth=int(Feature_selection['depth_of_trees'])))

    elif target['type'] == "classification":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectFromModel
        sel = SelectFromModel(RandomForestClassifier(n_estimators=int(Feature_selection['num_of_trees']), max_depth=int(Feature_selection['depth_of_trees'])))

    sel.fit(X, y)
    feature_importance = sel.estimator_.feature_importances_
    import numpy as np

    sorted_indices = np.argsort(feature_importance)[::-1]
    keep_columns = data.columns[np.concatenate((sorted_indices[:int(Feature_selection['num_of_features_to_keep'])], [list(data.columns).index(target_column)]))]
    data = data[keep_columns]

elif Feature_selection['feature_reduction_method'] == "No Reduction":
    pass

elif Feature_selection['feature_reduction_method'] == "Correlation with target":
    corr = data.corr()[target_column].drop(target_column)
    sorted_cor = sorted(dict(abs(corr).items()).items(), key=lambda x: x[1], reverse=True)[:int(Feature_selection['num_of_features_to_keep'])]
    keep_columns = np.array([key for key, value in sorted_cor] + [target_column])
    data = data[keep_columns]

elif Feature_selection['feature_reduction_method'] == "Principal Component Analysis":
    from sklearn.decomposition import PCA

    pca = PCA(n_components=int(Feature_selection['num_of_features_to_keep']))
    pca.fit(X)
    print(f"Explained variance: {pca.explained_variance_ratio_}")
    X = pca.transform(X)

else:
    AssertionError(f"Unknown feature reduction method: {Feature_selection['feature_reduction_method']}")

algorithms = Input['design_state_data']['algorithms']


def model_impliment(algo_name, hyperparameters):
    model = None
    model_name = hyperparameters.pop('model_name')

    if model_name == "Decision Tree":
        from sklearn.tree import DecisionTreeRegressor
        
        parameters = {
            'criterion': ['mse'] if hyperparameters["use_gini"] else ['mae'],  # Criterion for split: mse for Gini, mae for entropy
            'max_depth': range(hyperparameters["min_depth"], hyperparameters["max_depth"] + 1),
            'min_samples_leaf': hyperparameters["min_samples_per_leaf"],
            'splitter': ['best'] if hyperparameters["use_best"] else ['random'],  # Split strategy: best or random
            'random_state': [None] if hyperparameters["use_random"] else [0]  # Random state: None for random, 0 otherwise
        }
        given_model = DecisionTreeRegressor()

    elif model_name == "Random Forest Regressor":
        from sklearn.ensemble import RandomForestRegressor
        parameters = {
            'n_estimators': [hyperparameters["min_trees"], hyperparameters["max_trees"]],
            'max_depth': [hyperparameters["min_depth"], hyperparameters["max_depth"]],
            'min_samples_leaf': [hyperparameters["min_samples_per_leaf_min_value"], hyperparameters["min_samples_per_leaf_max_value"]]
            }
        given_model = RandomForestRegressor()

    elif model_name == "Random Forest Classifier":
        from sklearn.ensemble import RandomForestClassifier

        parameters = {
            'n_estimators': np.linspace(hyperparameters["min_trees"], hyperparameters["max_trees"], num=10, dtype=int),
            'max_features': ['auto', 'sqrt', 'log2'] if hyperparameters["feature_sampling_statergy"] == "Default" else ['auto'],
            'max_depth': np.linspace(hyperparameters["min_depth"], hyperparameters["max_depth"], num=10, dtype=int),
            'min_samples_leaf': np.linspace(hyperparameters["min_samples_per_leaf_min_value"], hyperparameters["min_samples_per_leaf_max_value"], num=10, dtype=int),
            'n_jobs': [hyperparameters["parallelism"]]
        }
        given_model = RandomForestClassifier()
    elif model_name == "LinearRegression":
        from sklearn.linear_model import LinearRegression

        parameters = {
            'fit_intercept': [True, False],
            'normalize': [True, False],
            'n_jobs': [hyperparameters["parallelism"]],
            'min_iter': [hyperparameters["min_iter"]],
            'max_iter': [hyperparameters["max_iter"]],
            'min_regparam': [hyperparameters["min_regparam"]],
            'max_regparam': [hyperparameters["max_regparam"]],
            'min_elasticnet': [hyperparameters["min_elasticnet"]],
            'max_elasticnet': [hyperparameters["max_elasticnet"]]
        }
        given_model = LinearRegression()
    
    elif model_name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression

        parameters = {
            'penalty': ['l1', 'l2'],
            'C': np.linspace(hyperparameters["min_regparam"], hyperparameters["max_regparam"], num=10),
            'solver': ['liblinear'],
            'max_iter': [hyperparameters["min_iter"], hyperparameters["max_iter"]],
            'n_jobs': [hyperparameters["parallelism"]]
        }
        given_model = LogisticRegression()
        4
    elif model_name == "RidgeRegression":
        from sklearn.linear_model import Ridge
        parameters = {
            'alpha': np.linspace(hyperparameters["min_regparam"], hyperparameters["max_regparam"], num=10),
            'max_iter': [hyperparameters["min_iter"], hyperparameters["max_iter"]]
        }
        given_model = Ridge()
    elif model_name == "GBTClassifier":
        from sklearn.ensemble import GradientBoostingClassifier

        parameters = {
            'n_estimators': np.linspace(hyperparameters["num_of_BoostingStages"][0], hyperparameters["num_of_BoostingStages"][1], num=10, dtype=int),
            'subsample': np.linspace(hyperparameters["min_subsample"], hyperparameters["max_subsample"], num=10),
            'learning_rate': hyperparameters["learningRate"],
            'max_depth': np.linspace(hyperparameters["min_depth"], hyperparameters["max_depth"], num=10, dtype=int),
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['auto', 'sqrt', 'log2'] if hyperparameters["feature_sampling_statergy"] == "Fixed number" else ['auto'],
            'verbose': [0]
        }
        given_model = GradientBoostingClassifier(loss='deviance' if hyperparameters["use_deviance"] else 'exponential', warm_start=False)

    elif model_name == "GBTRegressor":
        from sklearn.ensemble import GradientBoostingRegressor

        parameters = {
            'n_estimators': np.linspace(hyperparameters["num_of_BoostingStages"][0], hyperparameters["num_of_BoostingStages"][1], num=10, dtype=int),
            'subsample': np.linspace(hyperparameters["min_subsample"], hyperparameters["max_subsample"], num=10),
            'learning_rate': hyperparameters["learningRate"],
            'max_depth': np.linspace(hyperparameters["min_depth"], hyperparameters["max_depth"], num=10, dtype=int),
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['auto', 'sqrt', 'log2'] if hyperparameters["feature_sampling_statergy"] == "Fixed number" else ['auto'],
            'verbose': [0]
        }
        given_model = GradientBoostingRegressor(loss='deviance' if hyperparameters["use_deviance"] else 'exponential', warm_start=False)

    elif model_name == "XG Boost":

        import xgboost as xgb

        parameters = {
            'use_gradient_boosted_tree': [True],
            'dart': [True],
            'tree_method': ['auto'],  
            'random_state': [hyperparameters["random_state"]],
            'n_estimators': [hyperparameters["max_num_of_trees"]],
            'early_stopping_rounds': [hyperparameters["early_stopping_rounds"]],
            'max_depth': hyperparameters["max_depth_of_tree"],
            'learning_rate': hyperparameters["learningRate"],
            'reg_alpha': hyperparameters["l1_regularization"],
            'reg_lambda': hyperparameters["l2_regularization"],
            'gamma': hyperparameters["gamma"],
            'min_child_weight': hyperparameters["min_child_weight"],
            'subsample': hyperparameters["sub_sample"],
            'colsample_bytree': hyperparameters["col_sample_by_tree"],
            'missing': [None] if hyperparameters["replace_missing_values"] else [np.nan],
            'n_jobs': [hyperparameters["parallelism"]]
        }
        given_model = xgb.XGBClassifier() 
    
    elif model_name == "Decision Tree":
        from sklearn.tree import DecisionTreeClassifier
        
        parameters = {
            'criterion': ['gini'] if hyperparameters["use_gini"] else ['entropy'],
            'max_depth': range(hyperparameters["min_depth"], hyperparameters["max_depth"] + 1),
            'min_samples_leaf': hyperparameters["min_samples_per_leaf"],
            'splitter': ['best'] if hyperparameters["use_best"] else ['random'],
            'random_state': [None] if hyperparameters["use_random"] else [0]
        }
        given_model = DecisionTreeClassifier()
    elif model_name == "Support Vector Machine":
        from sklearn.svm import SVC
        
        parameters = {
            'kernel': ['linear'] if hyperparameters["linear_kernel"] else [],
            'kernel': ['rbf'] if hyperparameters["rep_kernel"] else [],
            'kernel': ['poly'] if hyperparameters["polynomial_kernel"] else [],
            'kernel': ['sigmoid'] if hyperparameters["sigmoid_kernel"] else [],
            'C': np.linspace(hyperparameters["c_value"][0], hyperparameters["c_value"][1], num=10),
            'shrinking': [True] if hyperparameters["auto"] else [False],
            'probability': [True] if hyperparameters["scale"] else [False],
            'gamma': ['auto'] if hyperparameters["auto"] else ['scale'] if hyperparameters["scale"] else np.linspace(0.0, 1.0, num=10),
            'tol': [hyperparameters["tolerance"]],
            'max_iter': [hyperparameters["max_iterations"]]
        }
        given_model = SVC()
    elif model_name == "Stochastic Gradient Descent":
        from sklearn.linear_model import SGDClassifier, SGDRegressor

        parameters = {
            'loss': ['log'] if hyperparameters["use_logistics"] else ['modified_huber'] if hyperparameters["use_modified_hubber_loss"] else [],
            'max_iter': [hyperparameters["max_iterations"]] if hyperparameters["max_iterations"] else [],
            'tol': [hyperparameters["tolerance"]],
            'penalty': ['l1', 'l2', 'elasticnet'] if hyperparameters["use_l1_regularization"] == "on" or hyperparameters["use_l2_regularization"] == "on" or hyperparameters["use_elastic_net_regularization"] else [],
            'alpha': np.linspace(hyperparameters["alpha_value"][0], hyperparameters["alpha_value"][1], num=10),
            'n_jobs': [hyperparameters["parallelism"]]
        }
        given_model = SGDClassifier() if target['type'] == "classification" else SGDRegressor()

    elif model_name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

        parameters = {
            'n_neighbors': hyperparameters["k_value"],
            'weights': ['uniform', 'distance'] if hyperparameters["distance_weighting"] else ['uniform'],
            'algorithm': ['auto'] if hyperparameters["neighbour_finding_algorithm"] == "Automatic" else [hyperparameters["neighbour_finding_algorithm"]],
            'p': [hyperparameters["p_value"]],
            'n_jobs': [1]  # Adjust according to available resources
        }
        given_model = KNeighborsClassifier() if target['type'] == "classification" else SGDRegressor()
    elif model_name == "Extra Random Trees":
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        parameters = {
            'n_estimators': np.linspace(hyperparameters["num_of_trees"][0], hyperparameters["num_of_trees"][1], num=10, dtype=int),
            'max_features': ['auto', 'sqrt', 'log2'] if hyperparameters["feature_sampling_statergy"] == "Square root and Logarithm" else ['auto'],
            'max_depth': np.linspace(hyperparameters["max_depth"][0], hyperparameters["max_depth"][1], num=10, dtype=int),
            'min_samples_leaf': np.linspace(hyperparameters["min_samples_per_leaf"][0], hyperparameters["min_samples_per_leaf"][1], num=10, dtype=int),
            'n_jobs': [hyperparameters["parallelism"]]
        }
        given_model = ExtraTreesClassifier() if target['type'] == "classification" else ExtraTreesRegressor()
    elif model_name == "Neural Network":
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        parameters = {
            'hidden_layer_sizes': [(layer,) for layer in range(hyperparameters["hidden_layer_sizes"][0], hyperparameters["hidden_layer_sizes"][1]+1)],
            'activation': ['relu', 'logistic', 'tanh'] if hyperparameters["activation"] == " " else [hyperparameters["activation"]],
            'alpha': [hyperparameters["alpha_value"]],
            'max_iter': [hyperparameters["max_iterations"]],
            'tol': [hyperparameters["convergence_tolerance"]],
            'solver': ['adam'] if hyperparameters["solver"] == "ADAM" else [hyperparameters["solver"]],
            'shuffle': [True] if hyperparameters["shuffle_data"] else [False],
            'learning_rate_init': [hyperparameters["initial_learning_rate"]],
            'batch_size': ['auto'] if hyperparameters["automatic_batching"] else [None],
            'beta_1': [hyperparameters["beta_1"]],
            'beta_2': [hyperparameters["beta_2"]],
            'epsilon': [hyperparameters["epsilon"]],
            'power_t': [hyperparameters["power_t"]],
            'momentum': [hyperparameters["momentum"]],
            'nesterovs_momentum': [hyperparameters["use_nesterov_momentum"]]
        }
        given_model = MLPClassifier() if target['type'] == "classification" else MLPRegressor()
    else:
        print("Given Model is not in our Library")

    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(given_model, parameters, cv=5, n_jobs=-1)
    model.fit(X, y)
    print(f"Best parameters: {model.best_params_}")
    print(f"Best score: {model.best_score_}")
    return model.best_estimator_


print("\n")


for algorithm_given, parameter_given in algorithms.items():
    is_selected = parameter_given.pop('is_selected')
    model_name = parameter_given['model_name']

    if not is_selected:
        continue

    model = model_impliment(algorithm_given, parameter_given)
    print(model_name)
    pretty_print(parameter_given)
    
    if not model is None:
        break

print("Thank You")
