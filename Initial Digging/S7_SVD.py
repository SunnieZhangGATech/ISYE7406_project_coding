from sklearn.decomposition import TruncatedSVD

from S0_prepare_data_models import DATASET_MODE, prepare_data, create_models
from public_func import  cv_training_models, data_split, export_model_scores, training_models

def run_all_models_cv(data_path, output_folder, title_name, ds_mode=DATASET_MODE.tf_idf_plus_numerical):
    X, y = prepare_data(data_path, ds_mode=ds_mode)
    X_columns = X.columns
    print(X_columns)

    dimensions = X.shape
    print(f"X Features has {dimensions[0]} rows and {dimensions[1]} columns.")

    print("Perform SVD...")
    theme = 'SVD'
    overall_report = {'Components':[], 'Models': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F-1 score': [], 'ROC_AUC Score':[], 'Elapsed Time':[]}
    for n in range(2,32,5):
        print(f"SVD Components={n}")

        svd = TruncatedSVD(n_components=n)
        X_svd = svd.fit_transform(X)

        models = create_models()
        X_train, X_test, y_train, y_test = data_split(X_svd, y, test_size=0.2)

        training_models(X_train, X_test, y_train, y_test, models, f'{title_name} n={n} ', output_folder + theme + f'_n_{n}_')

        report = cv_training_models(X_svd, y, models, f'{title_name} n={n} ', output_folder + theme + f'_n_{n}_')
        for key, vals in report.items():
            overall_report[key].extend(vals)
        overall_report['Components'].extend([n for _ in report['Models']])

    export_model_scores(overall_report, output_folder + theme + '_evaluation' + ".csv")

data_path = "data/US_Accidents_cleaned.csv"

print("########### numerical only ####################")
output_folder = "output/SVD/num_"
run_all_models_cv(data_path, output_folder, title_name='SVD Numerical Only', ds_mode=DATASET_MODE.numerical_only)

print("########### description only ####################")
output_folder = "output/SVD/Desc_"
run_all_models_cv(data_path, output_folder, title_name='SVD Description Only', ds_mode=DATASET_MODE.tf_idf_only)

print("########### numerical + description ####################")
output_folder = "output/SVD/comb_"
run_all_models_cv(data_path, output_folder, title_name='SVD Numerical + Description', ds_mode=DATASET_MODE.tf_idf_plus_numerical)
