import numpy as np

from S0_prepare_data_models import DATASET_MODE, prepare_data, create_models
from public_func import cv_training_models, data_split, export_model_scores, training_models, mutual_information_feature_selection

def run_all_models_cv(data_path, output_folder, title_name, ds_mode=DATASET_MODE.tf_idf_plus_numerical):
    X, y = prepare_data(data_path, ds_mode=ds_mode)
    X_columns = X.columns
    print(X_columns)

    dimensions = X.shape
    print(f"X Features has {dimensions[0]} rows and {dimensions[1]} columns.")

    print("Perform Mutual Information...")
    theme = 'MI'
    overall_report = {'alpha':[], 'Models': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F-1 score': [], 'ROC_AUC Score':[], 'Elapsed Time':[]}
    for n in range(1, 31, 5):
        alpha = n/1000
        print(f"Mutual Information alpha={alpha}")

        mi_features = mutual_information_feature_selection(X, y, alpha, X_columns, output_folder + theme + f'_alpha_{alpha}.png')
        del_features = [i for i, col in enumerate(X_columns) if col not in mi_features]
        X_mi = np.delete(X, del_features, 1)
        keep_features = [i for i, col in enumerate(X_columns) if col in mi_features]
        print("keep_features: ", len(keep_features), keep_features)
        # mi_features = mutual_information_feature_selection(X, y, alpha, X_columns, output_folder + theme + f'_alpha_{alpha}.png')
        # keep_features = [i for i, col in enumerate(X_columns) if col in mi_features]
        # print("keep_features: ", len(keep_features), keep_features)
        # X_mi = X[:, keep_features]

        models = create_models()
        X_train, X_test, y_train, y_test = data_split(X_mi, y, test_size=0.2)

        #  23  PercentSalaryHike         1470 non-null   int64
        #  24  PerformanceRating         1470 non-null   int64
        training_models(X_train, X_test, y_train, y_test, models, f"{title_name} alpha={alpha}", output_folder + theme + f'_alpha_{alpha}_')

        report = cv_training_models(X_mi, y, models, f"{title_name} alpha={alpha}", output_folder + theme + f'_alpha_{alpha}_')
        for key, vals in report.items():
            overall_report[key].extend(vals)
        overall_report['alpha'].extend([alpha for _ in report['Models']])

    export_model_scores(overall_report, output_folder + theme + '_evaluation' + ".csv")

data_path = "data/US_Accidents_cleaned.csv"

print("########### numerical only ####################")
output_folder = "output/MI/num_"
run_all_models_cv(data_path, output_folder, title_name='MI Numerical Only', ds_mode=DATASET_MODE.numerical_only)

print("########### description only ####################")
output_folder = "output/MI/Desc_"
run_all_models_cv(data_path, output_folder, title_name='MI Description Only', ds_mode=DATASET_MODE.tf_idf_only)

print("########### numerical + description ####################")
output_folder = "output/MI/comb_"
run_all_models_cv(data_path, output_folder, title_name='MI Numerical + Description', ds_mode=DATASET_MODE.tf_idf_plus_numerical)
