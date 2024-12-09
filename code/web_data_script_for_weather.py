import pandas as pd
import requests
import os
import time
import json
from tqdm import tqdm

for year in range(2018, 2025):
    new_folder_path = f"weather/{year}"
    station_airports = f"weather/station-airports/{year}"
    failedLogsPath = f"weather/{year}/failed-files"
    os.makedirs(new_folder_path, exist_ok=True)
    os.makedirs(failedLogsPath, exist_ok=True)
    os.makedirs(station_airports, exist_ok=True)

    # airport data preprocessing
    print("Start preprocessing all airports data in the US.", flush=True)
    file_path = 'airports_with_timezone.csv'
    airport_data = pd.read_csv(file_path)

    # Selecting and rounding latitude and longitude to 3 decimal places, using only the first occurrence for each airport.
    unique_airports = airport_data.drop_duplicates(subset="AIRPORT_ID").copy()
    unique_airports = unique_airports.drop(unique_airports.index[-1])
    unique_airports["LATITUDE"] = unique_airports["LATITUDE"].round(3)
    unique_airports["LONGITUDE"] = unique_airports["LONGITUDE"].round(3)
    
    # try:
    #     current_time = datetime.utcnow()

    #     def get_utc_offset(timezone_name):
    #         try:
    #             timezone = pytz.timezone(timezone_name)
    #             offset = timezone.utcoffset(current_time).total_seconds() / 3600
    #             return offset
    #         except pytz.UnknownTimeZoneError:
    #             print(f"Unknown time zone: {timezone_name}")
    #             return None

    #     unique_airports['UTC_OFFSET'] = unique_airports['TIMEZONE'].apply(get_utc_offset)
    # except Exception as e:
    #     print(f"Error occurred during timezone conversion: {e}")
    
    # Reducing the dataset to just relevant columns
    result = unique_airports[["AIRPORT_ID", "LONGITUDE", "LATITUDE", "timezone"]]
    result.to_csv(f"{station_airports}/all_airports.csv", index=False)
    print(f"Airport data preprocessing is done. {station_airports}/all_airports.csv\n", flush=True)
    # exit()

    # # download stations data
    # url = f"https://www.ncei.noaa.gov/access/services/search/v1/data?startDate={year}-01-01T00:00:00&endDate={year}-12-31T23:59:59&bbox=71.351,-178.217,18.925,179.769&place=Country:117&dataset=local-climatological-data-v2&limit=9999"

    # max_retries = 5
    # retry_delay = 5

    # print("Start downloading the metadata of all stations in the US.", flush=True)
    # for attempt in range(max_retries):
    #     try:
    #         response = requests.get(url, timeout=10) 
    #         if response.status_code == 200:
    #             data = response.json()
    #             with open(f"{station_airports}/all_stations.json", 'w') as f:
    #                 print(f"Saving data to local file... (./weather/{year}/all_stations.json)", flush=True)
    #                 json.dump(data, f)

    #             print("Successfully downloaded the metadata of all stations in the US.", flush=True)
    #             with open(f"{station_airports}/all_stations.csv", 'w') as f:
    #                 print(f"Saving data to local file... (./weather/{year}/all_stations.csv)", flush=True)
    #                 f.write("id,longitude,latitude,filesize\n")
                    
    #                 #  tqdm for station 
    #                 for i, item in enumerate(tqdm(data['results'], desc="Processing Stations", unit="station")):
    #                     station_id = item['stations'][0]['id'] if item['stations'] else None
    #                     coordinates = item['location']['coordinates']
    #                     fileSize = item['fileSize']
    #                     f.write(f"{station_id},{coordinates[0]},{coordinates[1]},{fileSize}\n")
    #             break
    #         else:
    #             print(f"Error: {response.status_code}, retrying...")
    #     except Exception as e:
    #         print(f"Exception occurred: {e}, retrying...")

    #     time.sleep(retry_delay)
    # else:
    #     print("Failed to download the metadata after multiple attempts. Restart the program.")
    #     exit()    

print("Done!")















# import pandas as pd
# import requests
# import os
# import time
# import json


# for year in range(2019,2025):
# # year = 2018
#   new_folder_path = f"data/{year}"
#   station_airports = f"data/station-airports/{year}"
#   failedLogsPath = f"data/{year}/failed-files"
#   os.makedirs(new_folder_path, exist_ok=True)
#   os.makedirs(failedLogsPath, exist_ok=True)
#   os.makedirs(station_airports, exist_ok=True)

#   # airport data preprocessing
#   print("Start preprocessing all airports data in the US.",flush=True)
#   file_path = 'Airport.csv'
#   airport_data = pd.read_csv(file_path)

#   # Selecting and rounding latitude and longitude to 3 decimal places, using only the first occurrence for each airport.
#   unique_airports = airport_data.drop_duplicates(subset="AIRPORT_ID").copy()
#   unique_airports["LATITUDE"] = unique_airports["LATITUDE"].round(3)
#   unique_airports["LONGITUDE"] = unique_airports["LONGITUDE"].round(3)

#   # Reducing the dataset to just relevant columns
#   result = unique_airports[["AIRPORT_ID", "LONGITUDE","LATITUDE"]]
#   result.to_csv(f"{station_airports}/all_airports.csv", index=False)
#   print(f"Airport data preprocessing is done. {station_airports}/all_airports.csv\n",flush=True)

#   # download data
#   url = f"https://www.ncei.noaa.gov/access/services/search/v1/data?startDate={year}-01-01T00:00:00&endDate={year}-12-31T23:59:59&bbox=71.351,-178.217,18.925,179.769&place=Country:117&dataset=local-climatological-data-v2&limit=9999"

#   max_retries = 5
#   retry_delay = 5

#   print("Start downloading the metadata of all stations in the US.",flush=True)
#   for attempt in range(max_retries):
#     try:
#       response = requests.get(url, timeout=10) 
#       if response.status_code == 200:
#         data = response.json()
#         with open(f"{station_airports}/all_stations.json", 'w') as f:
#           print(f"Saving data to local file... (./data/{year}/all_airports.json)",flush=True)
#           json.dump(data, f)
          
#         print("Successfully downloaded the metadata of all airpots in the US.",flush=True)
#         with open(f"{station_airports}/all_stations.csv", 'w') as f:
#           print(f"Saving data to local file... (./data/{year}/all_stations.csv)",flush=True)
#           f.write(f"id,longitude,latitude\n")
#           for i in range(len(data['results'])):
#             print(f"{i + 1} / {len(data['results'])}, {(i + 1) / len(data['results']) * 100:.2f}%",flush=True)
#             item = data['results'][i]
#             station_id = item['stations'][0]['id'] if item['stations'] else None
#             coordinates = item['location']['coordinates']
#             f.write(f"{station_id},{coordinates[0]},{coordinates[1]}\n")
#         break
#       else:
#         print(f"Error: {response.status_code}, retrying...")
#     except Exception as e:
#       print(f"Exception occurred: {e}, retrying...")

#     time.sleep(retry_delay)
#   else:
#     print("Failed to download the metadata after multiple attempts. Restart the program.")
#     exit()

# print("Done!")
