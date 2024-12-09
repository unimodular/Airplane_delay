import pandas as pd
from datetime import datetime, timedelta

years = range(2023, 2025)
months = [1, 11, 12]

for year in years:
    print(f"Merging data for year {year}")
    
    map_df = pd.read_csv(f'weather/station-airports/{year}/nearest_stations_{year}.csv').drop(columns=['Distance (miles)','Filesize'])
    
    for month in months:
        if year == 2024 and month != 1:
            break
        
        print(f"month {month}")
        
        flights_df = pd.read_csv(f'flight/On_Time_Marketing_Carrier_On_Time_Performance_(Beginning_January_2018)_{year}_{month}.csv', low_memory=False)
    
        # Merge flights with UTC offset information
        merged_df = flights_df.merge(map_df, 
                          left_on='OriginAirportID', 
                          right_on='Airport ID', 
                          how='left')
        
        del flights_df
        
        # merged_df = merged_df.dropna(subset=['DepTime'])
        
        date_time_str = merged_df['FlightDate'] + merged_df['CRSDepTime'].astype(str).str.zfill(4)
        merged_df['FlightPlanTimeStr'] = pd.to_datetime(date_time_str, format="%Y-%m-%d%H%M")
        merged_df = merged_df.dropna(subset=['AirportTimezone'])
        merged_df['FlightPlanUTC'] = merged_df.apply(
            lambda row: row['FlightPlanTimeStr']
            .tz_localize(row['AirportTimezone'], ambiguous=True)
            .tz_convert('UTC'),
            axis=1
        )
        
        # date_time_str = merged_df['FlightDate'] + merged_df['DepTime'].astype(int).astype(str).str.zfill(4)
        # date_time_str.to_csv('date_time_str1.csv')
        # merged_df['FlightActualTimeStr'] = pd.to_datetime(date_time_str, format="%Y-%m-%d%H%M", errors='coerce')
        # merged_df['FlightActualTimeStr'] = pd.to_datetime(date_time_str, format="%Y-%m-%d%H%M")
        # merged_df.to_csv('temp.csv')
        # exit()
        
        # merged_df['FlightActualUTC'] = merged_df.apply(
        #     lambda row: row['FlightActualTimeStr']
        #     .tz_localize(row['AirportTimezone'])
        #     .tz_convert('UTC'),
        #     axis=1
        # )
      
        output_csv_path = f'merged_{year}_{month}.csv'
        write_header = True
        
        for station_id, group in merged_df.groupby('Nearest Station ID'):
            station_timezone = group['StationTimezone'].iloc[0]
            try:
                station_df = pd.read_csv(f'weather/{year}/stations-data/LCD_{station_id}_{year}.csv')
            except FileNotFoundError as error_message:
                print(error_message)
            else:
                station_df['StationTimeStr'] = pd.to_datetime(station_df['DATE'])
                station_df['StationUTC'] = station_df['StationTimeStr'].dt.tz_localize(station_timezone, ambiguous=False).dt.tz_convert('UTC')
                
                group.sort_values(by=['FlightPlanUTC'], inplace=True)
                df_merged = pd.merge_asof(group, station_df, left_on='FlightPlanUTC', right_on='StationUTC', direction='backward', suffixes=('', '_plr1'))
        
                # Step 2: Compute the target time for r2 (1 hour earlier than r1)
                df_merged['target_time_pl'] = df_merged['StationUTC'] - pd.Timedelta(hours=1)
        
                df_merged = df_merged.dropna(subset=('target_time_pl'))
                # Step 3: Find r2 using merge_asof with direction 'nearest'
                df_merged = pd.merge_asof(
                    df_merged, station_df, left_on='target_time_pl', right_on='StationUTC', 
                    direction='nearest', suffixes=('', '_plr2'), tolerance=pd.Timedelta('10 minutes')
                )
                
                
                # df_merged.sort_values(by=['FlightActualUTC'], inplace=True)
                # df_merged = pd.merge_asof(df_merged, station_df, left_on='FlightActualUTC', right_on='StationUTC', direction='backward', suffixes=('', '_acr1'))
        
                # # Step 2: Compute the target time for r2 (1 hour earlier than r1)
                # df_merged['target_time_ac'] = df_merged['StationUTC_acr1'] - pd.Timedelta(hours=1)
        
                # # Step 3: Find r2 using merge_asof with direction 'nearest'
                # df_merged = pd.merge_asof(
                #     df_merged, station_df, left_on='target_time_ac', right_on='StationUTC', 
                #     direction='nearest', suffixes=('', '_acr2'), tolerance=pd.Timedelta('1 minutes')
                # )
                df_merged.to_csv(output_csv_path, mode='a', index=False, header=write_header)
                write_header = False

