import numpy as np

from S0_prepare_data_models import DATASET_MODE, prepare_data, create_models
from public_func import rf_feature_importance, cv_training_models, data_split, \
    export_model_scores, training_models, Lasso_feature_coefficient


def run_all_models_cv(data_path, output_folder, title_name, ds_mode=DATASET_MODE.tf_idf_plus_numerical):
    X, y = prepare_data(data_path, ds_mode=ds_mode)
    # X_columns = [f'V{i}' for i in range(X.shape[1])]
    X_columns = X.columns
    print(X_columns)
    print(y)

    dimensions = X.shape
    print(f"X Features has {dimensions[0]} rows and {dimensions[1]} columns.")

    rf_feature_importance(X, y, X_columns, output_folder + 'Random_Forest_feature_importance.png')

    print("Perform Lasso...")
    theme = 'Lasso'
    overall_report = {'alpha': [], 'Models': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F-1 score': [], 'ROC_AUC Score':[], 'Elapsed Time': []}

    for n in range(1, 21, 5):
        alpha = n/1000
        print(f"Lasso alpha={alpha}")

        lasso_features = Lasso_feature_coefficient(X, y, alpha, X_columns, output_folder + theme + f'_alpha_{alpha}_feature_coefficient.png')
        del_features = [i for i, col in enumerate(X_columns) if col not in lasso_features]
        X_lasso = np.delete(X, del_features, 1)
        print("Lasso features: ", lasso_features, X_lasso.shape)

        models = create_models()
        X_train, X_test, y_train, y_test = data_split(X_lasso, y, test_size=0.2)

        #  23  PercentSalaryHike         1470 non-null   int64
        #  24  PerformanceRating         1470 non-null   int64
        training_models(X_train, X_test, y_train, y_test, models, f"{title_name} alpha={alpha}", output_folder + theme + f'_alpha_{alpha}_')

        report = cv_training_models(X_lasso, y, models, f"{title_name} alpha={alpha}", output_folder + theme + f'_alpha_{alpha}_')
        for key, vals in report.items():
            overall_report[key].extend(vals)
        overall_report['alpha'].extend([alpha for _ in report['Models']])

    export_model_scores(overall_report, output_folder + theme + '_evaluation' + ".csv")

data_path = "data/US_Accidents_cleaned.csv"

# print("########### numerical only ####################")
# output_folder = "output/Lasso/num_"
# run_all_models_cv(data_path, output_folder, title_name='Lasso Numerical Only', ds_mode=DATASET_MODE.numerical_only)
#
# print("########### description only ####################")
# output_folder = "output/Lasso/Desc_"
# run_all_models_cv(data_path, output_folder, title_name='Lasso Description Only', ds_mode=DATASET_MODE.tf_idf_only)

print("########### numerical + description ####################")
output_folder = "output/Lasso/comb_"
run_all_models_cv(data_path, output_folder, title_name='Lasso Numerical + Description', ds_mode=DATASET_MODE.tf_idf_plus_numerical)
