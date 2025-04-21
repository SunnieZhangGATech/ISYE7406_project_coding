import numpy as np
from sklearn.decomposition import PCA

from S0_prepare_data_models import DATASET_MODE, prepare_data, create_models
from public_func import cv_training_models, data_split, export_model_scores, training_models, plot_pca_loadings

def run_all_models_cv(data_path, output_folder, title_name, ds_mode=DATASET_MODE.tf_idf_plus_numerical):
    X, y = prepare_data(data_path, ds_mode=ds_mode)
    X_columns = X.columns
    print(X_columns)

    dimensions = X.shape
    print(f"X Features has {dimensions[0]} rows and {dimensions[1]} columns.")

    print("Perform PCA...")
    theme = f'{title_name} PCA'
    overall_report = {'Components':[], 'Models': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F-1 score': [], 'ROC_AUC Score':[], 'Elapsed Time':[]}
    for n in range(2, 20, 5):
        print(f"PCA Components={n}")

        pca = PCA(svd_solver='arpack', n_components=n)
        X_pca = pca.fit_transform(X)

        # Sum the loadings of the first two principal components
        loadings = np.abs(pca.components_)
        explained_ratio = np.abs(pca.explained_variance_ratio_)
        loading_sums = np.sum(loadings.T * explained_ratio, axis=1)
        # loading_sums = np.sum(np.abs(loadings[:n, :]), axis=0)
        plot_pca_loadings(loading_sums, X_columns, output_folder + f'n_{n}_plot_loadings.png')

        # # explained variance
        # explained_variance_ratio(X_pca, output_folder + theme + f'_n_{n}_explained_variance.png')

        models = create_models()
        X_train, X_test, y_train, y_test = data_split(X_pca, y, test_size=0.2)

        training_models(X_train, X_test, y_train, y_test, models, f'{theme} n={n} ', output_folder + theme + f'_n_{n}_')

        report = cv_training_models(X_pca, y, models, f'{theme} n={n} ', output_folder + theme + f'_n_{n}_')
        for key, vals in report.items():
            overall_report[key].extend(vals)
        overall_report['Components'].extend([n for _ in report['Models']])

    export_model_scores(overall_report, output_folder + theme + '_evaluation' + ".csv")

data_path = "data/US_Accidents_cleaned.csv"

print("########### numerical only ####################")
output_folder = "output/PCA/num_"
run_all_models_cv(data_path, output_folder, title_name='PCA Numerical Only', ds_mode=DATASET_MODE.numerical_only)

print("########### description only ####################")
output_folder = "output/PCA/Desc_"
run_all_models_cv(data_path, output_folder, title_name='PCA Description Only', ds_mode=DATASET_MODE.tf_idf_only)

print("########### numerical + description ####################")
output_folder = "output/PCA/comb_"
run_all_models_cv(data_path, output_folder, title_name='PCA Numerical + Description', ds_mode=DATASET_MODE.tf_idf_plus_numerical)
