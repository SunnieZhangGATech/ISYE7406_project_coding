from enum import Enum

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb

# encountered the following errors, had to switch to TruncatedSVD
# numpy.core._exceptions._ArrayMemoryError: Unable to allocate 90.1 GiB for an array with shape (303282, 39892) and data type float64
# TypeError: PCA only support sparse inputs with the "arpack" solver, while "auto" was passed. See TruncatedSVD for a possible alternative.
# def build_PCA_X(data, X_addition=None):
#     from sklearn.decomposition import PCA
#     from scipy.sparse import csr_matrix
#
#     # Vectorize text data
#     vectorizer = TfidfVectorizer(stop_words='english')
#     X_text = vectorizer.fit_transform(data['Description'])
#
#     # Convert text data to DataFrame
#     # df_text = pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out())
#     # Convert to sparse matrix
#     df_text = csr_matrix(X_text)
#
#     # Apply PCA to reduce dimensionality
#     pca = PCA(n_components=10)  # Adjust n_components as needed
#     X_text_reduced = pca.fit_transform(df_text)
#
#     # Convert reduced text data to DataFrame
#     X_text_df_reduced = pd.DataFrame(X_text_reduced, columns=[f'V{i}' for i in range(X_text_reduced.shape[1])])
#
#     # Combine reduced text features with other features
#     if X_addition is None:
#         X = X_text_df_reduced
#     else:
#         X = pd.concat([X_addition.reset_index(drop=True), X_text_df_reduced], axis=1)
#
#     return X

def create_models():
    ########### time-consuming models ####################
    #  SVM, stopped for long run, more than 3 hours
    #  Gradient Boosting, stopped for long run, more than 2 hours

    print("----- Creating models --------")
    models = []

    # Model: Logistic Regression
    print("**** Logistic Regression ****")
    # SAGA:
    # Use Case: Suitable for large-scale datasets and sparse data. It can handle both dense and sparse input efficiently.
    # Multi-class: Supports multinomial loss for multi-class classification.
    model = LogisticRegression(multi_class='multinomial', solver='sag', max_iter=1000)
    models.append( ('Logistic Regression', model) )

    # Model: KNN
    # Create a K-Nearest Neighbors model (with 5 neighbors, you can adjust this value)
    print("**** K-Nearest Neighbors (KNN) ****")
    model = KNeighborsClassifier()
    models.append( ('KNN', model) )

    # Model: SVM
    # SVM is too time-consuming
    # print("**** Support Vector Machines (SVM) ****")
    # model = SVC(kernel='rbf', probability=True)
    # models.append( ('SVM', model) )

    # Model: DecisionTreeClassifier
    print("**** Decision Trees ****")
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    models.append( ('Decision Trees', model) )

    # Model: Random Forest
    print("**** Random Forest ****")
    model = RandomForestClassifier(n_estimators=15, max_depth=5, random_state=42)
    models.append( ('Random Forest', model) )

    # Model: Gradient Boosting
    # Gradient Boosting can be time-consuming for multi-class classification, especially with large datasets and complex models.
    # print("**** Gradient Boosting & XGBoost ****")
    # model = GradientBoostingClassifier()
    # models.append( ('Gradient Boosting', model) )

    # LightGBM
    print("**** LightGBM ****")
    # # Initialize the LightGBM model with verbose=-1 to suppress logging
    model = lgb.LGBMClassifier(objective='multiclass', num_class=4, verbose=-1)
    models.append( ('LightGBM', model) )

    # XGBoost
    print("**** XGBoost ****")
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=4, eval_metric='mlogloss')
    models.append( ('XGBoost', model) )

    # # Model: AdaBoost
    # print("**** AdaBoost ****")
    # model = AdaBoostClassifier(algorithm='SAMME', n_estimators=50, learning_rate=1.0)
    # models.append( ('AdaBoost', model) )

    # Model: Neural Networks (time-consuming)
    print("**** Neural Networks ****")
    model = MLPClassifier(max_iter=1000)
    models.append( ('Neural Networks', model) )

    print("------------------------------------")

    return models

def build_TF_IDF_X(data, svd_components=500, X_addition=None):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=5) #, max_features=3000
    tfidf_matrix = vectorizer.fit_transform(data['Description'].str.lower())
    print(f"TfidfVectorizer size: {tfidf_matrix.shape}")

    # Apply TruncatedSVD to reduce dimensionality
    svd = TruncatedSVD(n_components=svd_components)  # Adjust n_components as needed
    X_text_svd = svd.fit_transform(tfidf_matrix)

    # Convert reduced text data to DataFrame
    # X_text_reduced = pd.DataFrame(X_text_svd, columns=[f'V_{i}' for i in range(X_text_svd.shape[1])])
    X_text_reduced = pd.DataFrame(X_text_svd, columns=[f'SVD{i}' for i in range(X_text_svd.shape[1])])
    # X_text = pd.DataFrame(X_text_svd.toarray(), columns=vectorizer.get_feature_names_out())

    # Combine reduced text features with other features
    if X_addition is None:
        X = X_text_reduced
    else:
        X = pd.concat([X_text_reduced, X_addition.reset_index(drop=True)], axis=1)

    return X

# class syntax
class DATASET_MODE(Enum):
    numerical_only = 1
    tf_idf_only = 2
    tf_idf_plus_numerical = 3

# extract new features from the accident description using TF-IDF vectorization,
# perform dimensionality reduction using TruncatedSVD
# then combine them with the numerical features for classification.
def prepare_data(data_path, ds_mode=DATASET_MODE.tf_idf_plus_numerical, svd_components=500):

    # Load dataset
    data = pd.read_csv(data_path)

    if ds_mode == DATASET_MODE.tf_idf_only:
        print('+++++ data mode: description only +++++' )
        X = build_TF_IDF_X(data, svd_components)
    else:
        # Preprocessing numerical data
        X_numerical = data.drop(['Description', 'Severity'], axis=1)
        # X_numerical.columns = X_numerical.columns.astype(str)

        # Initialize scalers
        standard_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()

        # Fit and transform the data using StandardScaler
        standard_scaled_numerical = pd.DataFrame(standard_scaler.fit_transform(X_numerical), columns=X_numerical.columns)

        # Fit and transform the standardized data using MinMaxScaler
        scaled_numerical = pd.DataFrame(minmax_scaler.fit_transform(standard_scaled_numerical), columns=X_numerical.columns)

        # # Standardize the data
        # scaler = StandardScaler()
        # scaled_numerical = pd.DataFrame(scaler.fit_transform(X_numerical), columns=X_numerical.columns)

        if ds_mode == DATASET_MODE.numerical_only:
            print('+++++ data mode: numerical only +++++')
            X = scaled_numerical.reset_index(drop=True)
        else:
            print('+++++ data mode: Description + numerical +++++' )
            X = build_TF_IDF_X(data, svd_components=svd_components,  X_addition=scaled_numerical)

    # print("\n### X sample ###")
    # print(X.head())
    #
    # print("\n### X info ###")
    # print(X.info())
    #
    # print("\n### X Describe ###")
    # print(X.describe())

    # # Convert column names to strings
    # X_text_df = pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out())

    # # Combine text features with other features
    # X = pd.concat([X_text_df, X_other.reset_index(drop=True)], axis=1)
    y = data['Severity']
    # Encode the target labels to start from 0
    # XGBoost expects the class labels to start from 0 and be consecutive integers.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # print("\n### Y sample ###")
    # print(y.head())
    #
    # print("\n### Y info ###")
    # print(y.info())
    #
    # print("\n### Y Describe ###")
    # print(y.describe())

    return X, y_encoded

# extreme long run time
def perform_RF(X_train, y_train, X_test, y_test):
    # Train model
    print("\n### Train model ...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predict
    print("\n### Predicting ...")
    y_pred = model.predict(X_test)

    # Evaluate
    print("\n### Evaluating ...")
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

def perform_lgb(X_train, y_train, X_test, y_test):
    import lightgbm as lgb
    # Train model with LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {'objective': 'multiclass', 'num_class': 5, 'metric': 'multi_logloss'}
    model = lgb.train(params, train_data, num_boost_round=100)

    # Predict
    y_pred = model.predict(X_test)
    y_pred = [list(x).index(max(x)) for x in y_pred]

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

def test_lgb():
    #data_path = "data/US_Accidents_sampled_500k.csv"
    data_path = "data/US_Accidents_cleaned.csv"

    X, y = prepare_data(data_path, DATASET_MODE.numerical_only)
    # X, y = prepare_data(data_path, DATASET_TYPE.tf_idf_only)
    # X, y = prepare_data(data_path, DATASET_TYPE.tf_idf_plus_numerical)

    # Split data
    print("\n### Split data ...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #perform_RF(X_train, y_train, X_test, y_test)
    perform_lgb(X_train, y_train, X_test, y_test)

# TF-IDF + numeric features: Accuracy: 93.8%/93.53%/
# TF-IDF only: Accuracy: 92.4% (1000), 92.5% (200),
# numeric features only : Accuracy: 90.72% 89.71%

# test_lgb()