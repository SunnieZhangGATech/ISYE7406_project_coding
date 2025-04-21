from public_func import load_data, sample_dataset, subset_year

def sample_data_year(year=2022):
    data_path = "data/US_Accidents_March23.csv"
    data = load_data(data_path)

    # # drop columns
    # columns = ['ID', 'Description', 'Distance(mi)', 'End_Time', 'End_Lat', 'End_Lng', 'Country', 'Turning_Loop']
    # data = data.drop(columns, axis=1)

    subset_year(data, year=year, file_path=f"data/US_Accidents_{year}.csv")

def sample_data():
    data_path = "data/US_Accidents_2022.csv"
    data = load_data(data_path, True)
    sample_dataset(data, fraction=1/700, file_path="data/US_Accidents_sample.csv")

def sample_data2():
    data_path = "data/US_Accidents_sampled_500k.csv"
    data = load_data(data_path, True)
    sample_dataset(data, fraction=1/10, file_path="data/US_Accidents_sampled_50k.csv")

# sample_data_year(2023)
# sample_data()
sample_data2()
