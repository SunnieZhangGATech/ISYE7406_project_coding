import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD

from S0_prepare_data_models import DATASET_MODE, prepare_data, create_models
from public_func import rf_feature_importance, cv_training_models, explained_variance_ratio, \
    Lasso_feature_coefficient, mutual_information_feature_selection

def run_all_models_cv(data_path, output_folder, title_name='', ds_mode=DATASET_MODE.tf_idf_plus_numerical):
    X, y = prepare_data(data_path, ds_mode=ds_mode)
    X_columns = X.columns
    print(X_columns)

    # choose most important features

    dimensions = X.shape
    print(f"X Features has {dimensions[0]} rows and {dimensions[1]} columns.")

    # # Standardize the data
    # # should categorical data be standardized?
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    # # Encode the target labels to start from 0
    # # XGBoost expects the class labels to start from 0 and be consecutive integers.
    # label_encoder = LabelEncoder()
    # y_encoded = label_encoder.fit_transform(y)

    rf_feature_importance(X, y, X_columns, output_folder + 'RF_feature_importance.png')

    model_report = cv_training_models(X, y, create_models(), title_name, output_folder, cv=5)

    print("############## model score report ###################")
    print(model_report)

    # PCA
    n = 4
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)
    # explained variance
    explained_variance_ratio(X, output_folder + f'n_{n}_explained_variance.png')
    cv_training_models(X_pca, y, create_models(), f'PCA n={n} ', output_folder + f'PCA_n_{n}_')

    # Lasso
    alpha = 0.001
    print(f"Lasso alpha={alpha}")
    lasso_features = Lasso_feature_coefficient(X, y, alpha, X_columns, output_folder + f'Lasso_alpha_{alpha}_.png')
    del_features = [i for i, col in enumerate(X_columns) if col not in lasso_features]
    X_lasso = np.delete(X, del_features, 1)
    print("Lasso features: ", lasso_features, X_lasso.shape)
    cv_training_models(X_lasso, y, create_models(), f"Lasso alpha={alpha}", output_folder + f'Lasso_alpha_{alpha}_')

    # Mutual Information
    alpha = 0.006
    print(f"Mutual Information alpha={alpha}")
    mi_features = mutual_information_feature_selection(X, y, alpha, X_columns, output_folder + f'mutual_information_alpha_{alpha}.png')
    del_features = [i for i, col in enumerate(X_columns) if col not in mi_features]
    X_mi = np.delete(X, del_features, 1)
    print("Mutual Information features: ", mi_features, X.shape)

    cv_training_models(X_mi, y, create_models(), f"Mutual Information alpha={alpha}", output_folder + f'MI_alpha_{alpha}_')

    # SVD
    n = 30
    svd = TruncatedSVD(n_components=n)
    X_svd = svd.fit_transform(X)
    cv_training_models(X_svd, y, create_models(), f'SVD n={n} ', output_folder + f'SVD_n_{n}_')

data_path = "data/US_Accidents_cleaned.csv"

# print("########### numerical only ####################")
output_folder = "output/models/num/"
run_all_models_cv(data_path, output_folder, title_name=' Numerical Only', ds_mode=DATASET_MODE.numerical_only)

print("########### description only ####################")
output_folder = "output/models/desc/"
run_all_models_cv(data_path, output_folder, title_name=' Description Only', ds_mode=DATASET_MODE.tf_idf_only)

# print("########### numerical + description ####################")
output_folder = "output/models/"
run_all_models_cv(data_path, output_folder, title_name=' Numerical + Description', ds_mode=DATASET_MODE.tf_idf_plus_numerical)


