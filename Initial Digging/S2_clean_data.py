import pandas as pd
from public_func import load_data, convert_numerical, remove_outliers_IsolationForest

def data_clean(data_path, output_path):
    print("cleaning data...")

    data = load_data(data_path, True)

    # Create a new 'Year' column
    # data['Year'] = pd.to_datetime(data['Start_Time'], format='mixed').dt.year
    # data['Month'] = pd.to_datetime(data['Start_Time'], format='mixed').dt.month
    # data['Day'] = pd.to_datetime(data['Start_Time'], format='mixed').dt.day
    data['DayOfWeek'] = pd.to_datetime(data['Start_Time'], format='mixed').dt.dayofweek
    data['HourOfDay'] = pd.to_datetime(data['Start_Time'], format='mixed').dt.hour

    # ID	Source	Severity	Start_Time	End_Time	Start_Lat	Start_Lng	End_Lat	End_Lng	Distance(mi)	Description	Street	City	County	State	Zipcode	Country	Timezone	Airport_Code	Weather_Timestamp	Temperature(F)	Wind_Chill(F)	Humidity(%)	Pressure(in)	Visibility(mi)	Wind_Direction	Wind_Speed(mph)	Precipitation(in)	Weather_Condition	Amenity	Bump	Crossing	Give_Way	Junction	No_Exit	Railway	Roundabout	Station	Stop	Traffic_Calming	Traffic_Signal	Turning_Loop	Sunrise_Sunset	Civil_Twilight	Nautical_Twilight	Astronomical_Twilight
    # drop columns
    # columns = ['ID', 'Description', 'Distance(mi)', 'End_Time', 'End_Lat', 'End_Lng', 'Country', 'Turning_Loop']
    columns = ['ID', 'Source', 'Start_Time', 'End_Time', 'End_Lat', 'End_Lng', 'Country', 'Turning_Loop', 'Weather_Timestamp']
    data = data.drop(columns, axis=1)

    # Remove rows with any missing values
    data = data.dropna()

    # convert Severity to binary value
    # new_data = data.copy()
    # new_data['Severity'] = data['Severity'] > 2
    # data = data.drop('Severity', axis=1)
    # data = data.rename(columns={'SevereAccident': 'Severity'})

    # new_data = convert_numerical(data, exclusive='Description')

    # Convert all columns to numerical except column 'Description'
    new_data = convert_numerical(data.drop(columns=['Description']))

    # outliers
    clean_data = remove_outliers_IsolationForest(new_data)
    #clean_data = remove_outliers_IQR(data, ratio=1.5)

    # add column Description
    clean_data['Description'] = data['Description']

    print("\n### cleaned data info ###")
    print(clean_data.info())

    clean_data.to_csv(output_path, index=False)

    print("data was cleaned...")

# data_path = "data/US_Accidents_2022.csv"
# data_path = "data/US_Accidents_sampled_500k.csv"
data_path = "data/US_Accidents_sampled_50k.csv"
# data_path = "data/US_Accidents_sampled_5k.csv"
output_path = "data/US_Accidents_cleaned.csv"

data_clean(data_path, output_path)
