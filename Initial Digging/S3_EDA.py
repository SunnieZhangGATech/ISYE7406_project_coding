from public_func import load_data, visualize_response, convert_numerical, Boruta_selection
from public_func import plot_pies_in_one, boxplot_in_one, histogram_in_one, correlation_matrix

# data_path = "data/US_Accidents_March23.csv"
# output_folder = "output/RAW_EDA/"

def perform_EDA(data_path, output_folder=None, drop_columns=None, perform_Boruta = False):
    data = load_data(data_path)

    if output_folder:
        visualize_response(data, fd_name='Severity', ff_path=output_folder + "pie_Severity.png")

    # drop columns
    if drop_columns is None:
        drop_columns = ['ID', 'Description', 'Distance(mi)', 'End_Time', 'End_Lat', 'End_Lng', 'Country','Turning_Loop']
    if len(drop_columns) > 0:
        data = data.drop(drop_columns, axis=1)

    # # Create a new 'Year' column
    # data['Year'] = pd.to_datetime(data['Start_Time'], format='mixed').dt.year
    # data['Month'] = pd.to_datetime(data['Start_Time'], format='mixed').dt.month

    data = convert_numerical(data)

    # outliers detection
    # if output_folder:
    #     num_cols = data.select_dtypes(exclude=['object'])
    #     for col in num_cols:
    #         print("col_name: ", col)
    #         boxplot_outliers(data, col, output_folder + f'box_plot_{col}.png')
    #         histogram_outliers(data, col, output_folder + f'histogram_{col}.png')

    # Selecting all columns except the target
    X_org = data.drop('Severity', axis=1)
    # Selecting the target column
    y = data['Severity']

    print("Plot inone...")
    if output_folder:
        # boxplot_outliers_in_one(X_org, output_folder + f'box_plot.png')
        plot_pies_in_one(X_org, exclusive=['Start_Time', 'Start_Lat','Start_Lng','Description'], num_rows=7, num_cols=5, ff_path=output_folder + f'pie_plot.png')
        boxplot_in_one(X_org, num_rows=5, num_cols=8, ff_path=output_folder + f'box_plot.png')
        histogram_in_one(X_org, num_rows=8, num_cols=5, ff_path=output_folder + f'histogram.png')
        correlation_matrix(X_org, y, ff_path=output_folder + 'correlation_matrix.png')

    if perform_Boruta:
        print("Determines feature importance using Boruta Algorithm")
        selected_features, rejected_features = Boruta_selection(X_org, y, ff_path=output_folder + f'boruta.png')
        print("Selected features:", selected_features)
        print("Rejected features:", rejected_features)

    print("EDA was done!")

def raw_data_EDA():
    data_path = "data/US_Accidents_sampled_10k.csv"
    output_folder = "output/EDA/"
    perform_EDA(data_path,output_folder, drop_columns=[], perform_Boruta=False)

def cleaned_data_EDA():
    data_path = "data/US_Accidents_cleaned.csv"
    # data_path = "data/US_Accidents_sampled_500k.csv"
    output_folder = "output/CLN_EDA/"
    perform_EDA(data_path,output_folder, drop_columns=[], perform_Boruta=True)

raw_data_EDA()
cleaned_data_EDA()
