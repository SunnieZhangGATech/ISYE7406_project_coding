# US Accidents (2016 - 2023)

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, RepeatedStratifiedKFold, \
    StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    roc_curve, auc, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Lasso
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
from boruta import BorutaPy


def load_data(data_path, simple_EDA=False):
    df = pd.read_csv(data_path)
    print("###  US_Accidents ###")
    print(df.shape)

    if simple_EDA:
        print("\n### Data sample ###")
        print(df.head())

        print("\n### Data info ###")
        print(df.info())

        print("\n### Data Describe ###")
        print(df.describe())

        print("\n### Missing Values ###")
        # print(df.describe())
        df.isnull().sum()

    return df

# Function to randomly sample 10% of a dataset
def sample_dataset(data, fraction, file_path):
    ds = data.sample(frac=fraction)
    ds.to_csv(file_path, index=False)

def subset_year(data, year, file_path):
    ds = data[pd.to_datetime(data['Start_Time'], format='mixed').dt.year == year]
    ds.to_csv(file_path, index=False)

# Visulization
def visualize_response(data, fd_name, ff_path):
    Severity_counts = data[fd_name].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(Severity_counts, labels=Severity_counts.index, autopct='%1.1f%%',startangle=140)
    plt.title(f'{fd_name} Distribution')
    plt.axis('equal')
    plt.savefig(ff_path)
    plt.close()

def detect_outliers(data, col_name, ff_path):
    data = np.array([10, 12, 12, 13, 12, 11, 14, 13, 100])
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(x - mean) / std for x in data]
    outliers = [x for x in data if np.abs((x - mean) / std) > 3]
    print(outliers)

def boxplot_together(data, num_rows, num_cols, ff_path):
    # Create a figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))

    # Flatten the axes array to iterate through subplots easily
    axes_flat = axes.flatten()

    # Iterate through the DataFrame columns and plot histograms with distinct colors
    for i, (column, ax) in enumerate(zip(data.columns, axes_flat)):
        sns.boxplot(ax=ax, data=data[column])
        ax.set_title(column)

    # Adjust layout and display the plot
    plt.title('Box Plot')
    plt.tight_layout()
    plt.savefig(ff_path)
    plt.close()

def boxplot_in_one(data, num_rows, num_cols, ff_path):
    # scaler = StandardScaler()
    # data_scaled = scaler.fit_transform(data)
    # df_scaled = pd.DataFrame(data_scaled, columns=data.columns)

    # Create a figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 20))

    # Flatten the axes array to iterate through subplots easily
    axes_flat = axes.flatten()

    # Get a list of (16) distinct colors from the tab20 colormap
    colors = plt.cm.tab20.colors + plt.cm.tab20.colors + plt.cm.tab20.colors
    # print(colors)

    # Iterate through the DataFrame columns and plot histograms with distinct colors
    for i, (column, ax) in enumerate(zip(data.columns, axes_flat)):
        ax.boxplot(data[column], patch_artist=True, boxprops=dict(facecolor=colors[i], color='black'))
        ax.set_title(column)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig(ff_path)
    plt.close()

def boxplot_in_one00(data, num_rows, ff_path):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    df_scaled = pd.DataFrame(data_scaled, columns=data.columns)

    df_split = np.array_split(df_scaled, num_rows, axis=1)

    # Create a figure and subplots
    fig, axes = plt.subplots(nrows=num_rows, figsize=(20, 20))

    # Flatten the axes array to iterate through subplots easily
    axes_flat = axes.flatten()

    # Iterate through the DataFrame columns and plot histograms with distinct colors
    for i, ax in enumerate(axes_flat):
        sns.boxplot(ax=ax, data=df_split[i])
        ax.tick_params(labelrotation=45)

    # Adjust layout and display the plot
    plt.title('Box Plot')
    plt.tight_layout()
    plt.savefig(ff_path)
    plt.close()

def boxplot_outliers(data, col_name, ff_path):
    plt.figure(figsize=(3, 5))
    plt.boxplot(data[col_name])
    plt.title(f'Box Plot - {col_name}')
    plt.tight_layout()
    # plt.show()
    plt.savefig(ff_path)
    plt.close()

def plot_pies_in_one(data, num_rows, num_cols, ff_path, max_slices = 10):
    # Create a figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(35, 20))

    # Flatten the axes array to iterate through subplots easily
    axes_flat = axes.flatten()

    # Get a list of (16) distinct colors from the tab20 colormap
    # colors = plt.cm.tab20.colors + plt.cm.tab20.colors + plt.cm.tab20.colors
    # print(colors)

    # Iterate through the DataFrame columns and plot histograms with distinct colors
    for i, (column, ax) in enumerate(zip(data.columns, axes_flat)):
        value_counts = data[column].value_counts()

        # Aggregate smaller slices into 'Other'
        if len(value_counts) > max_slices:
            value_counts = value_counts.head(max_slices)

        #val_counts = data[column].value_counts()
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.2f%%', startangle=140)
        ax.set_title(column)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig(ff_path)
    plt.close()

def histogram_in_one(data, num_rows, num_cols, ff_path, max_bins = 20):
    # Create a figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(35, 20))

    # Flatten the axes array to iterate through subplots easily
    axes_flat = axes.flatten()

    # Get a list of (16) distinct colors from the tab20 colormap
    colors = plt.cm.tab20.colors + plt.cm.tab20.colors + plt.cm.tab20.colors
    # print(colors)

    # Iterate through the DataFrame columns and plot histograms with distinct colors
    for i, (column, ax) in enumerate(zip(data.columns, axes_flat)):
        unique_values = data[column].nunique()
        if unique_values < max_bins:
            ax.hist(data[column], bins=unique_values, color=colors[i], edgecolor='black')
        else:
            ax.hist(data[column], bins=max_bins, color=colors[i], edgecolor='black')

        ax.set_title(column)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig(ff_path)
    plt.close()

def stacked_histogram_in_one(data1, data2, num_rows, num_cols, ff_path):
    # Create a figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 10))

    # Flatten the axes array to iterate through subplots easily
    axes_flat = axes.flatten()

    # Get a list of (16) distinct colors from the tab20 colormap
    colors = plt.cm.tab20.colors + plt.cm.tab20.colors
    # print(colors)

    # Iterate through the DataFrame columns and plot histograms with distinct colors
    for i, (column, ax) in enumerate(zip(data1.columns, axes_flat)):
        ax.hist([data1[column], data2[column]], stacked=True, density=True, edgecolor='black')
        ax.set_title(column)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig(ff_path)
    plt.close()

def histogram_outliers(data, col_name, ff_path, max_bins=20):
    unique_values = data[col_name].nunique()
    if unique_values < max_bins:
        plt.hist(data[col_name], bins=unique_values, color='blue', edgecolor='black')
    else:
        plt.hist(data[col_name], bins=max_bins, color='blue', edgecolor='black')
    plt.title(f'Histogram - {col_name}')
    plt.tight_layout()
    # plt.show()
    plt.savefig(ff_path)
    plt.close()

# Convert categorical data into numerical
def convert_numerical(data, exclusive=[]):
    print("Convert categorical data into numerical...")
    cat_cols = data.select_dtypes(include = ['object', 'bool'])
    print("Categorical fields: ", cat_cols)
    # Create a LabelEncoder object
    label_encoder = LabelEncoder()
    categorical_cols = cat_cols
    for col in categorical_cols:
        if col not in exclusive:
            data[col] = label_encoder.fit_transform(data[col])

    print("\n### Data sample ###")
    print(data.head())

    print("\n### Data info ###")
    print(data.info())

    return data

def plot_roc_curve_in_one(models,X_test, y_test, num_rows, num_cols, title_name, ff_path):
    # Create a figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))

    # Flatten the axes array to iterate through subplots easily
    axes_flat = axes.flatten()
    colors = plt.cm.tab20.colors + plt.cm.tab20.colors

    # Iterate through the DataFrame columns and plot histograms with distinct colors
    for i, ((model_name, model), ax) in enumerate(zip(models, axes_flat)):
        y_pred = model.predict(X_test)
        plot_ax_roc_curve(ax, colors[i], model_name, y_test, y_pred)

    # Adjust layout and display the plot
    fig.suptitle(title_name + ' ROC AUC')
    plt.tight_layout()
    plt.savefig(ff_path)
    plt.close()

def plot_confusion_matrix_in_one(models,X_test, y_test, num_rows, num_cols, title_name, ff_path):
    # Create a figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))

    # Flatten the axes array to iterate through subplots easily
    axes_flat = axes.flatten()

    # Iterate through the DataFrame columns and plot histograms with distinct colors
    for i, ((model_name, model), ax) in enumerate(zip(models, axes_flat)):
        y_pred = model.predict(X_test)
        plot_ax_confusion_matrix(ax, model_name, y_test, y_pred)

    # Adjust layout and display the plot
    fig.suptitle(title_name + ' Confusion Matrix')
    plt.tight_layout()
    plt.savefig(ff_path)
    plt.close()

def plot_ax_roc_curve(ax, color, model_name, y_test, y_pred):
    # Predict probabilities for the test set
    # y_prob = model.predict_proba(X_test)[:, 1]

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    ax.plot(fpr, tpr, color=color, lw=3, label='(AUC = %0.2f)' % roc_auc)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(model_name)
    ax.legend(loc="lower right")
    # plt.show()

def plot_ax_confusion_matrix(ax, model_name, y_test, y_pred):
    # Predict labels for the test set
    # y_pred = model.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    # names = ['TN', 'FP', 'FN', 'TP']
    # counts = [value for value in cm.flatten()]
    # percentages = ['{0:.2%}'.format(value) for value in cm.flatten() / np.sum(cm)]
    # labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]
    # labels = np.asarray(labels).reshape(2, 2)
    # sns.heatmap(cm, ax=ax, annot=labels, cmap='RdYlGn', fmt='')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    ax.set_title(model_name)
    # plt.title('Confusion Matrix')
    # plt.show()

def data_split(X, y, test_size=0.25):
    # Split dataset into Training and Testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=6742)
    # print(f"X_train shape: {X_train.shape}")
    # print(f"X_test shape: {X_test.shape}")
    # print(f"y_train shape: {y_train.shape}")
    # print(f"y_test shape: {y_test.shape}")

    # Scaling
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# def score_report0(y, y_pred):
#     precision = precision_score(y, y_pred, average="samples", zero_division=0)
#     recall = recall_score(y, y_pred, average="samples")
#     f1 = f1_score(y, y_pred, average="samples")
#
#     return {'Precision':precision, 'Recall':recall, 'F-1 score':f1}

def score_report(y, y_pred):
    accuracy = accuracy_score(y, y_pred)

    rri = len(y_pred[(y_pred == 1) & (y == 1)])
    # Precision = # retrieved relevant instances / # total retrieved
    precision = rri / sum(y_pred == 1)
    # Recall = # retrieved relevant instances / # total relevant
    recall = rri / sum(y == 1)
    # F1 score = 2/(1/Precision + 1/Recall)
    f1 = 2 * precision * recall / (precision + recall)
    # roc_auc = roc_auc_score(y, y_pred)

    return {'Accuracy': accuracy, 'Precision': precision, 'Recall':recall, 'F-1 score':f1}

def cv_classifier_model(model, X, y, cv=5):
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),    # 'macro'
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted')
    }

    print(f"Starting cross-validation...")
    # Record the start time
    start_time = time.time()

    # Create a stratified k-fold cross-validator
    stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_validate(model, X, y, scoring=scoring, cv=stratified_cv, return_train_score=True)

    # Record the end time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    avg_accuracy = scores['test_accuracy'].mean()
    avg_precision = scores['test_precision'].mean()
    avg_recall = scores['test_recall'].mean()
    avg_f1 = scores['test_f1'].mean()

    return {'Accuracy': avg_accuracy, 'Precision': avg_precision, 'Recall':avg_recall,
            'F-1 score':avg_f1}, elapsed_time

# def cv_classifier_model0(model, X, y, cv=5):
#     scoring = {
#         'accuracy': make_scorer(accuracy_score),
#         'precision': make_scorer(precision_score, average='macro'),
#         'recall': make_scorer(recall_score, average='macro'),
#         'f1': make_scorer(f1_score, average='macro'),
#         'roc_auc': make_scorer(roc_auc_score, multi_class='ovr')
#     }
#
#     scores = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)
#
#     avg_accuracy = scores['test_accuracy'].mean()
#     avg_precision = scores['test_precision'].mean()
#     avg_recall = scores['test_recall'].mean()
#     avg_f1 = scores['test_f1'].mean()
#     # avg_cv = scores['test_cross_valid'].mean()
#     avg_roc_auc = scores['test_roc_auc'].mean()
#
#     return {'Accuracy': avg_accuracy, 'Precision': avg_precision, 'Recall':avg_recall,
#             'F-1 score':avg_f1, 'ROC_AUC Score':avg_roc_auc}

def run_classifier_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    # Make predictions on the test data
    y_test_predict = model.predict(X_test)

    return score_report(y_train, y_train_predict), score_report(y_test, y_test_predict)

def run_XGBoost(X_train, X_test, y_train, y_test):
    # Create a DMatrix (XGBoost's data structure)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set parameters for XGBoost
    params = {
        'objective': 'multi:softmax',  # Specify multiclass classification
        'num_class': 3,  # Number of classes
        'max_depth': 3,
        'eta': 0.1,
        'eval_metric': 'mlogloss'
    }

    # Train the model
    bst = xgb.train(params, dtrain, num_boost_round=100)

    # Make predictions
    preds = bst.predict(dtest)

    # Evaluate the model
    accuracy = accuracy_score(y_test, preds)
    print(f'Accuracy: {accuracy:.2f}')

def add_2_model_report(model_report, model_name, score_report, elapsed_time):
    model_report['Models'].append(model_name)
    model_report['Accuracy'].append(score_report['Accuracy'])
    model_report['Precision'].append(score_report['Precision'])
    model_report['Recall'].append(score_report['Recall'])
    model_report['F-1 score'].append(score_report['F-1 score'])
    model_report['Elapsed Time'].append(elapsed_time)

    if 'ROC_AUC Score' in score_report:
        model_report['ROC_AUC Score'].append(score_report['ROC_AUC Score'])

def output_model_scores(model_report, report_name, file_path):
    plot_model_scores(model_report, report_name, file_path + ".png")
    plot_model_elapsed_time(model_report, report_name, file_path + "_time.png")
    export_model_scores(model_report, file_path + ".csv")

def export_model_scores(model_report, ff_path):
    df = pd.DataFrame.from_dict(model_report)
    df.to_csv(ff_path, index=False)

def plot_model_elapsed_time(model_report, title_name, ff_path):
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    print('Models: ', model_report['Models'])
    models = model_report['Models']

    plt.plot(models, model_report['Elapsed Time'], label='Elapsed Time')
    plt.scatter(models, model_report['Elapsed Time'])
    for i, val in enumerate(model_report['Elapsed Time']):
        plt.text(models[i], val, f'{round(val, 2)}', fontsize=12)

    # Add title and labels
    plt.title(title_name + ' Elapsed Time of Models')
    plt.xlabel('Models')
    plt.ylabel('Elapsed Time')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(ff_path)
    plt.close()

def plot_model_scores(model_report, title_name, ff_path):
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    print('Models: ', model_report['Models'])
    models = model_report['Models']

    plt.plot(models, model_report['Accuracy'], label='Accuracy')
    plt.scatter(models, model_report['Accuracy'])
    for i, val in enumerate(model_report['Accuracy']):
        plt.text(models[i], val, f'{round(val, 2)}', fontsize=12)

    plt.plot(models, model_report['Precision'], label='Precision')
    plt.scatter(models, model_report['Precision'])
    for i, val in enumerate(model_report['Precision']):
        plt.text(models[i], val, f'{round(val, 2)}', fontsize=12)

    plt.plot(models, model_report['Recall'], label='Recall')
    plt.scatter(models, model_report['Recall'])
    for i, val in enumerate(model_report['Recall']):
        plt.text(models[i], val, f'{round(val, 2)}', fontsize=12)

    plt.plot(models, model_report['F-1 score'], label='F-1 score')
    plt.scatter(models, model_report['F-1 score'])
    for i, val in enumerate(model_report['F-1 score']):
        plt.text(models[i], val, f'{round(val, 2)}', fontsize=12)

    if 'ROC_AUC Score' in model_report and len(model_report['ROC_AUC Score'])==len(models):
        plt.plot(models, model_report['ROC_AUC Score'], label='ROC_AUC Score')
        plt.scatter(models, model_report['ROC_AUC Score'])
        for i, val in enumerate(model_report['ROC_AUC Score']):
            plt.text(models[i], val, f'{round(val, 2)}', fontsize=12)

    # Add title and labels
    plt.title(title_name + ' Performance Scores of Models')
    plt.xlabel('Models')
    plt.ylabel('Performance Scores')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Show the plot
    # plt.show()
    plt.savefig(ff_path)
    plt.close()

def cv_training_models(X, y, title_name, output_folder, cv=5):
    # model_report = {'Models': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F-1 score': [], 'Cross Validation Score': [], 'ROC_AUC Score':[]}
    # model_report = {'Models': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F-1 score': [], 'ROC_AUC Score':[]}
    model_report = {'Models': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F-1 score': [], 'Elapsed Time':[]}

    models = create_models()
    for model_name, model in models:
        print(f"----- Performing {model_name} --------")
        cv_scores, elapsed_time = cv_classifier_model(model, X, y, cv=cv)

        print(cv_scores)
        add_2_model_report(model_report, model_name, cv_scores, elapsed_time)

    output_model_scores(model_report, title_name, output_folder + 'model_report_cv')

    return model_report

def training_models(X_train, X_test, y_train, y_test, title_name, output_folder):
    model_report = {'Models':[], 'Accuracy':[], 'Precision':[],'Recall':[], 'F-1 score':[], 'Elapsed Time':[]}
    train_model_report = {'Models':[], 'Accuracy':[], 'Precision':[], 'Recall':[], 'F-1 score':[], 'Elapsed Time':[]}

    models = create_models()
    for model_name, model in models:
        print(f"----- Performing {model_name} --------")
        # Record the start time
        start_time = time.time()

        train_scores, test_scores = run_classifier_model(model, X_train, X_test, y_train, y_test)
        # Record the end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        add_2_model_report(model_report, model_name, test_scores, elapsed_time)
        add_2_model_report(train_model_report, model_name, train_scores, elapsed_time)

    # plot_roc_curve_in_one(models, X_test, y_test, 2, 6, title_name, output_folder + 'roc_curve.png')
    plot_confusion_matrix_in_one(models, X_test, y_test, 2, 6, title_name, output_folder + 'confusion_matrix.png')

    output_model_scores(model_report, title_name, output_folder + 'model_report_test')
    output_model_scores(train_model_report, title_name, output_folder +'model_report_train')

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
    #
    # # Model: Neural Networks (time-consuming)
    # print("**** Neural Networks ****")
    # model = MLPClassifier(max_iter=1000)
    # models.append( ('Neural Networks', model) )

    print("------------------------------------")

    return models

def save_plot_confusion_matrix(model_name, y_test, y_pred, ff_path):
    plt.figure(figsize=(10, 6))

    plot_ax_confusion_matrix(plt.gca(), model_name, y_test, y_pred)

    plt.tight_layout()
    plt.savefig(ff_path)
    plt.close()

def plot_variances(var_variance, ff_path):
    # Step 2: Plot the variances
    plt.figure(figsize=(15, 10))
    bars = plt.bar(range(1,len(var_variance)+1), var_variance, color='skyblue')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), va='bottom', ha='center')

    plt.title('Variance of Features')
    plt.xlabel('Features')
    plt.ylabel('Variance')
    plt.xticks(range(1,len(var_variance)+1), rotation=45)
    # plt.grid(True)
    # plt.show()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(ff_path)
    plt.close()

def explained_variance_ratio(data, ff_path):
    # Perform PCA
    pca = PCA()
    pca.fit(data)

    # Extract explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Plot the explained variance ratio
    plt.figure(figsize=(15, 10))
    bars = plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), va='bottom', ha='center')
    plt.title('Explained Variance Ratio by PCA')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(ff_path)
    plt.close()
    # plt.show()

def rf_feature_importance(X, y, feature_names, ff_path):
    # Create a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=6740)
    model.fit(X, y)

    # Get feature importances
    importances = model.feature_importances_

    # Create a DataFrame for better visualization
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    print(feature_importances)

    # Plot feature_importances
    plt.figure(figsize=(15, 10))
    bars = plt.bar(feature_importances['Feature'], feature_importances['Importance'], alpha=0.7, align='center')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), va='bottom', ha='center')
    plt.title('RandomForest Features Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(ff_path)
    plt.close()
    # plt.show()

def Lasso_feature_coefficient(X, y, alpha, feature_names, ff_path):
    # Create and fit the Lasso model
    print("features:", feature_names)
    # print("Lasso_feature_coefficient:", X, y)
    lasso = Lasso()  # You can adjust the alpha parameter
    lasso.fit(X, y)

    # Get the coefficients
    coefficients = lasso.coef_
    print("coefficients:", coefficients)

    # Create a DataFrame for better visualization
    feature_coefficient = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    print(feature_coefficient)
    feature_coefficient = feature_coefficient[feature_coefficient['Coefficient'] != 0]
    feature_coefficient = feature_coefficient.sort_values(by='Coefficient', ascending=False)
    print(feature_coefficient)

    # Plot feature coefficient
    plt.figure(figsize=(15, 10))
    bars = plt.bar(feature_coefficient['Feature'], feature_coefficient['Coefficient'], alpha=0.7, align='center')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), va='bottom', ha='center')
    plt.title('Lasso Features Coefficient')
    plt.xlabel('Features')
    plt.ylabel('Coefficient')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(ff_path)
    plt.close()
    # plt.show()

    lasso = Lasso(alpha=alpha)  # You can adjust the alpha parameter
    lasso.fit(X, y)
    coefficients = lasso.coef_
    feature_coefficient = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    feature_coefficient = feature_coefficient[feature_coefficient['Coefficient'] != 0]

    return feature_coefficient['Feature'].tolist()

def mutual_information_feature_selection(X, y, alpha, feature_names, ff_path):
    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y)

    # Create a DataFrame for better visualization
    mi_df = pd.DataFrame({'Feature': feature_names, 'Mutual_Information': mi_scores})
    mi_df = mi_df.sort_values(by='Mutual_Information', ascending=False)

    # Plot Mutual Information
    plt.figure(figsize=(15, 10))
    bars = plt.bar(mi_df['Feature'], mi_df['Mutual_Information'], alpha=0.7, align='center')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), va='bottom', ha='center')
    plt.title('Mutual Information Feature Selection')
    plt.xlabel('Features')
    plt.ylabel('Mutual Information')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(ff_path)
    plt.close()
    # plt.show()

    mi_features = mi_df[mi_df['Mutual_Information'] > alpha]
    print(mi_features)

    return mi_features['Feature'].tolist()

def data_balancing_SMOTE(data):
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline

    cols = list(data.columns)
    cols.remove('Severity')

    over = SMOTE(sampling_strategy=0.85)
    # under = RandomUnderSampler(sampling_strategy=0.1)
    f1 = data.loc[:, cols]
    t1 = data.loc[:, 'Severity']

    steps = [('over', over)]
    pipeline = Pipeline(steps=steps)
    X_bal, y = pipeline.fit_resample(f1, t1)
    print("#### data_balancing ####")
    print("X=", X_bal.shape)
    print("y=", y.shape)

    return X_bal, y

def correlation_matrix(X, y, ff_path):
    corr = X.corrwith(y).sort_values(ascending=False).to_frame()
    corr.columns = ['Severity']
    plt.figure(figsize=(8, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.4, linecolor='black')
    plt.title('Correlation w.r.t Severity')
    # Ensure labels are not cut off
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(ff_path)
    plt.close()

# IV Analysis: Measures the predictive power of each feature.

# Boruta Algorithm: Determines feature importance using a Random Forest-based method.
def Boruta_selection(X, y, ff_path):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize RandomForestClassifier
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

    # Initialize Boruta
    boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)

    # Fit Boruta
    boruta_selector.fit(X_train.values, y_train.values)

    # Get the ranking of features
    feature_ranks = boruta_selector.ranking_
    feature_names = X_train.columns

    # Create a DataFrame for visualization
    feature_df = pd.DataFrame({'Feature': feature_names, 'Rank': feature_ranks})

    # Sort the DataFrame by rank
    feature_df = feature_df.sort_values(by='Rank')

    # Plot the feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(feature_df['Feature'], feature_df['Rank'], color='skyblue')
    plt.xlabel('Rank')
    plt.ylabel('Features')
    plt.title('Feature Importance Ranking by Boruta')
    plt.tight_layout()
    plt.savefig(ff_path)
    plt.close()

    # Get the selected features
    selected_features = X_train.columns[boruta_selector.support_].to_list()
    rejected_features = X_train.columns[~boruta_selector.support_].to_list()

    return selected_features, rejected_features

# 1, Z-score: This method is simple and effective for normally distributed data. However,
# it may not work well for skewed distributions or datasets with varying scales.
# 2, Interquartile Range (IQR): This method is robust to non-normal distributions and is easy to implement.
# It works well for datasets with a clear central tendency and is less affected by extreme values.
# 3, Isolation Forest: This method is effective for high-dimensional datasets and can handle complex data structures.
# It is particularly useful when you have a large dataset with many features.
# 4, Local Outlier Factor (LOF): This method is good for detecting local outliers in datasets with varying densities.
# It is useful when the dataset has clusters of different densities and you want to identify outliers relative
# to their local neighborhood.
def remove_outliers_IsolationForest(data):
    from sklearn.ensemble import IsolationForest
    predict_field = '__Predict_response__'

    # Initialize the Isolation Forest model
    iso_forest = IsolationForest(contamination=0.1, random_state=42)

    # Fit the model
    iso_forest.fit(data)

    # Predict the outliers
    data[predict_field] = iso_forest.predict(data)

    # Filter the outliers
    # outliers = data[data[response_field] == -1]
    clean_data = data[data[predict_field] != -1]

    return clean_data.drop(predict_field, axis=1)

# define the lower and upper bounds for outliers as 1.5 times the IQR below Q1 and above Q3
def remove_outliers_IQR(data, ratio=1.5):
    # Select only numeric columns for outlier detection
    numeric_data = data.select_dtypes(include=['number'])

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)

    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - ratio * IQR
    upper_bound = Q3 + ratio * IQR

    # Remove outliers
    clean_data = data[~((numeric_data < lower_bound) | (numeric_data > upper_bound)).any(axis=1)]

    return clean_data

# def prepare_data(data_path, drop_columns=None, balance_data=False, output_folder=None):
#     data = load_data(data_path)
#
#     # Create a new 'Year' column
#     # data['Year'] = pd.to_datetime(data['Start_Time'], format='mixed').dt.year
#
#     if output_folder:
#         visualize_response(data, fd_name='Severity', ff_path=output_folder + "pie_Severity.png")
#
#     if drop_columns is None:
#         drop_columns = ['ID', 'Description', 'Distance(mi)', 'End_Time', 'End_Lat', 'End_Lng', 'Country','Turning_Loop']
#
#     if len(drop_columns) > 0:
#         data = data.drop(drop_columns, axis=1)
#
#     data = convert_numerical(data)
#
#     # outliers detection
#     if output_folder:
#         num_cols = data.select_dtypes(exclude=['object'])
#         for col in num_cols:
#             boxplot_outliers(data, col, output_folder + f'box_plot_{col}.png')
#             histogram_outliers(data, col, output_folder + f'histogram_{col}.png')
#
#     # Selecting all columns except the target
#     X_org = data.drop('Severity', axis=1)
#     # Selecting the target column
#     y = data['Severity']
#
#     if output_folder:
#         # boxplot_outliers_in_one(X_org, output_folder + f'box_plot.png')
#         boxplot_in_one(X_org, num_rows=3, ff_path=output_folder + f'box_plot.png')
#         histogram_in_one(X_org, num_rows=5, num_cols=8, ff_path=output_folder + f'histogram.png')
#         correlation_matrix(X_org, y, output_folder + 'correlation_matrix.png')
#
#     if balance_data:
#         X_org, y = data_balancing_SMOTE(data)
#
#     # Standardize the data
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_org)
#
#     print("X.shape:", X_scaled.shape)
#     print("Columns: ", X_org.columns)
#     # print("X_scaled Columns: ", X_scaled.columns)
#
#     return X_scaled, y, X_org.columns

# Accuracy: The ratio of correctly predicted instances to the total instances.
# It's a basic measure but can be misleading if the classes are imbalanced.
#
# Precision: The ratio of true positive predictions to the total predicted positives.
# It indicates how many of the predicted positive instances are actually positive.
#
# Recall (Sensitivity): The ratio of true positive predictions to the total actual positives.
# It shows how well the model captures all the positive instances.
#
# F1 Score: The harmonic mean of precision and recall.
# It provides a balance between precision and recall, especially useful when you need to balance both concerns.
#
# Confusion Matrix: A table that shows the true positives, true negatives, false positives, and false negatives.
# It provides a comprehensive view of the model's performance.
#
# ROC-AUC (Receiver Operating Characteristic - Area Under Curve): A plot that shows the trade-off between true positive rate and false positive rate.
# The AUC score summarizes the plot and provides a single measure of overall performance.
#
# Log Loss: Measures the uncertainty of the predictions.
# It penalizes false classifications more when the model is confident about the wrong prediction.