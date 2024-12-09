import pandas as pd
from haversine import haversine, Unit
from tqdm import tqdm 
from timezonefinder import TimezoneFinder

years = range(2018, 2025)

for year in years:
    print(f"Processing data for year {year}")

    airport_data = pd.read_csv(f'weather/station-airports/{year}/all_airports.csv')
    station_data = pd.read_csv(f'weather/station-airports/{year}/all_stations.csv')


    tf = TimezoneFinder()
    airport_data['TIMEZONE'] = airport_data.apply(lambda row: tf.timezone_at(lng=row['LONGITUDE'], lat=row['LATITUDE']), axis=1)
    station_data['TIMEZONE'] = station_data.apply(lambda row: tf.timezone_at(lng=row['longitude'], lat=row['latitude']), axis=1)
    
    results = []

    for _, airport in tqdm(airport_data.iterrows(), total=airport_data.shape[0], desc=f"Processing Airports {year}"):
        airport_id = airport['AIRPORT_ID']
        airport_coords = (airport['LATITUDE'], airport['LONGITUDE'])
        min_distance = float('inf')
        nearest_station = None
        filesize = -1
        stationTimezone = ''

        for _, station in station_data.iterrows():
            station_id = station['id']
            station_coords = (station['latitude'], station['longitude'])

            distance = haversine(airport_coords, station_coords, unit=Unit.MILES)

            if distance < min_distance:
                min_distance = distance
                nearest_station = station_id
                filesize =  station['filesize']
                stationTimezone = station['TIMEZONE']
                
        results.append({'Airport ID': airport_id, 'Nearest Station ID': nearest_station, 'Distance (miles)': min_distance, \
            'Filesize': filesize, 'AirportTimezone': airport['TIMEZONE'], 'StationTimezone': stationTimezone})

    results_df = pd.DataFrame(results)


    results_df.to_csv(f'weather/station-airports/{year}/nearest_stations_{year}.csv', index=False)
    print(f"Complete, result saved as nearest_stations_{year}.csv\n")
