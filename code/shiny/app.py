from faicons import icon_svg
from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import requests
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
import joblib
from datetime import datetime, timedelta
import pytz
from dateutil import parser
import re
import itertools
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, SGDClassifier

# Load models and scalers
delay_model = joblib.load('ols_regressor_model.joblib')
cancel_model = joblib.load('sgd_classifier_model.joblib')
delay_scaler = joblib.load('scaler_model_delay.joblib')
delay_encoder = joblib.load('encoder_model_delay.joblib')
cancel_scaler = joblib.load('scaler_model.joblib')
cancel_encoder = joblib.load('encoder_model.joblib')

# Categorical variables values
all_airlines = ['AA', 'AS', 'B6', 'DL', 'F9', 'G4', 'HA', 'NK', 'UA', 'VX', 'WN']
all_days_of_month = [i for i in range(1,32)]
all_days_of_week = [1, 2, 3, 4, 5, 6, 7]
all_months = [1, 11, 12]
all_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
with open('all_origins.txt', 'r') as file:
    all_origins = [line.strip() for line in file]
with open('all_dests.txt', 'r') as file:
    all_dests = [line.strip() for line in file]

# Define UI
app_ui = ui.page_fillable(
    ui.tags.div(
        ui.h1("Holiday Season Flight Prediction", style="text-align: left; margin-top: 0;"),
        style="position: absolute; top: 10px; left: 10px; z-index: 1000;"
    ),
    
    ui.navset_card_pill(
        ui.nav_spacer(),
        ui.nav_panel(
            "Flight Prediction",
            "Prediction of Arrival Delays and Flight Cancellation Rates for the Next 7 Days",
            ui.page_sidebar(
                ui.sidebar(
                    # User Input Section
                    ui.input_numeric("year1", "Year (departure)", value=2024, min=2024),
                    ui.input_select("month1", "Month (departure)", choices=[1, 11, 12], selected=11),
                    ui.input_numeric("dayofmonth1", "Day of Month (departure)", value=15, min=1, max=31),
                    ui.input_numeric("dayofweek1", "Day of Week (departure)", value=5, min=1, max=7),
                    ui.input_select("marketing_airline1", "Marketing Airline Network", choices=all_airlines, selected='AA'),
                    ui.input_select("origin1", "Origin Airport", choices=all_origins, selected='ORD'),
                    ui.input_select("dest1", "Destination Airport", choices=all_dests, selected='LGA'),
                    ui.input_numeric("dep_time1", "Scheduled Departure Time (e.g. 1 means 00:01, 1405 means 14:05)", value=600, min=0, max=2359),
                    ui.input_numeric("arr_time1", "Scheduled Arrival Time (e.g. 1 means 00:01, 1405 means 14:05)", value=900, min=0, max=2359),
                    ui.input_action_button("predict_btn", "Predict", theme="gradient-blue-indigo")
                ),
                
                ui.output_text_verbatim("error_message_1"),
                
                ui.layout_column_wrap(
                    ui.value_box(
                        "Hourly DewPoint Temperature",
                        ui.output_text("HourlyDewPointTemperature"),
                        theme="gradient-blue-indigo",
                        showcase=icon_svg("droplet"),
                    ),
                    ui.value_box(
                        "Hourly DryBulb Temperature",
                        ui.output_text("HourlyDryBulbTemperature"),
                        theme="gradient-blue-indigo",
                        showcase=icon_svg("temperature-half"),
                    ),
                    ui.value_box(
                        "Hourly Precipitation",
                        ui.output_text("HourlyPrecipitation"),
                        theme="gradient-blue-indigo",
                        showcase=icon_svg("snowflake"),
                    ),
                    ui.value_box(
                        "Hourly Pressure Change",
                        ui.output_text("HourlyPressureChange"),
                        theme="gradient-blue-indigo",
                        showcase=icon_svg("scale-unbalanced"),
                    ),
                    ui.value_box(
                        "Hourly Relative Humidity",
                        ui.output_text("HourlyRelativeHumidity"),
                        theme="gradient-blue-indigo",
                        showcase=icon_svg("water"),
                    ),
                    ui.value_box(
                        "Hourly SeaLevel Pressure",
                        ui.output_text("HourlySeaLevelPressure"),
                        theme="gradient-blue-indigo",
                        showcase=icon_svg("fish"),
                    ),
                    ui.value_box(
                        "Hourly Visibility",
                        ui.output_text("HourlyVisibility"),
                        theme="gradient-blue-indigo",
                        showcase=icon_svg("smog"),
                    ),
                    ui.value_box(
                        "Hourly Wind Speed",
                        ui.output_text("HourlyWindSpeed"),
                        theme="gradient-blue-indigo",
                        showcase=icon_svg("wind"),
                    )
                ),
                
                ui.layout_column_wrap(
                    ui.value_box(
                        "Predicted Arrival Delay",
                        ui.output_text("PredictedArrivalDelay"),
                        theme="bg-gradient-blue-cyan",
                        showcase=icon_svg("plane-circle-exclamation"),
                    ),
                    ui.value_box(
                        "Predicted Cancellation Probability",
                        ui.output_text("PredictedCancellationProbability"),
                        theme="bg-gradient-blue-cyan",
                        showcase=icon_svg("plane-circle-xmark"),
                    )
                )
            )
        ),
        
        ui.nav_panel(
            "General Prediction",
            "General Prediction using Partial Variables",
            ui.page_sidebar(
                ui.sidebar(
                    # User Input Section
                    ui.input_select("origin2", "Origin Airport", choices=all_origins),
                    ui.input_select("dest2", "Destination Airport", choices=all_dests),
                    ui.input_text("year2", "Year (departure)", placeholder="Optional"),
                    ui.input_text("month2", "Month (departure)", placeholder="Optional"),
                    ui.input_text("dayofmonth2", "Day of Month (departure)", placeholder="Optional"),
                    ui.input_text("dayofweek2", "Day of Week (departure)", placeholder="Optional"),
                    ui.input_text("marketing_airline2", "Marketing Airline Network", placeholder="Optional"),
                    ui.input_text("dep_time2", "Scheduled Departure Time (e.g. 1 means 00:01, 1405 means 14:05)", placeholder="Optional"),
                    ui.input_text("arr_time2", "Scheduled Arrival Time (e.g. 1 means 00:01, 1405 means 14:05)", placeholder="Optional"),
                    ui.input_text("hdpt", 'Hourly DewPoint Temperature', placeholder="Optional"),
                    ui.input_text("hdbt", 'Hourly DryBulb Temperature', placeholder="Optional"),
                    ui.input_text("hp", 'Hourly Precipitation', placeholder="Optional"),
                    ui.input_text("hpc", 'Hourly Pressure Change', placeholder="Optional"),
                    ui.input_text("hrh", 'Hourly Relative Humidity', placeholder="Optional"),
                    ui.input_text("hsp", 'Hourly SeaLevel Pressure', placeholder="Optional"),
                    ui.input_text("hv", 'Hourly Visibility', placeholder="Optional"),
                    ui.input_text("hws", 'Hourly Wind Speed', placeholder="Optional"),
                    ui.input_action_button("general_predict_btn", "Predict")
                ),
                
                ui.output_text_verbatim("error_message_2"),
                
                ui.layout_column_wrap(
                    ui.value_box(
                        "Average Predicted Delay",
                        ui.output_text("AveragePredictedDelay"),
                        theme="bg-gradient-cyan-green",
                        showcase=icon_svg("clock"),
                    ),
                    ui.value_box(
                        "Minimum Predicted Delay",
                        ui.output_text("MinimumPredictedDelay"),
                        theme="bg-gradient-cyan-green",
                        showcase=icon_svg("clock"),
                    ),
                    ui.value_box(
                        "Maximum Predicted Delay",
                        ui.output_text("MaximumPredictedDelay"),
                        theme="bg-gradient-cyan-green",
                        showcase=icon_svg("clock"),
                    ),
                ),
                
                ui.layout_column_wrap(
                    ui.value_box(
                        "Average Cancellation Probability",
                        ui.output_text("AverageCancellationProbability"),
                        theme="bg-gradient-cyan-green",
                        showcase=icon_svg("plane-slash"),
                    ),
                    ui.value_box(
                        "Minimum Cancellation Probability",
                        ui.output_text("MinimumCancellationProbability"),
                        theme="bg-gradient-cyan-green",
                        showcase=icon_svg("plane-slash"),
                    ),
                    ui.value_box(
                        "Maximum Cancellation Probability",
                        ui.output_text("MaximumCancellationProbability"),
                        theme="bg-gradient-cyan-green",
                        showcase=icon_svg("plane-slash"),
                    ),
                ),
                
                ui.value_box(
                    "Suggestions",
                    ui.output_text("suggestions"),
                    theme="bg-gradient-cyan-green",
                    showcase=icon_svg("lightbulb"),
                )
            )
        ),
            
        ui.nav_panel(
            "Statistical Data",
            "Presentation and Comparison of Airline and Airport Statistical Data",
            ui.page_sidebar(
                ui.sidebar(
                    # User Input Section for airline and airport selection
                    ui.input_selectize("selected_airlines", "Select Airlines", choices=all_airlines, multiple=True, options={"plugins": ["clear_button"]}),
                    ui.input_selectize("selected_origins", "Select Origin Airports", choices=all_origins, multiple=True, options={"plugins": ["clear_button"]}),
                    ui.input_selectize("selected_dests", "Select Destination Airports", choices=all_dests, multiple=True, options={"plugins": ["clear_button"]})
                ),
                
                ui.card(
                    ui.layout_columns(
                        ui.card(output_widget("airline_delay_plot")),
                        ui.card(output_widget("airline_cancel_plot"))
                    ),
                    ui.layout_columns(
                        ui.card(output_widget("origin_delay_plot")),
                        ui.card(output_widget("origin_cancel_plot"))
                    ),
                    ui.layout_columns(
                        ui.card(output_widget("dest_delay_plot")),
                        ui.card(output_widget("dest_cancel_plot"))
                    )
                )
            )
        )
    ),
    
    ui.card(
        ui.tags.footer(
            "Data source: https://www.transtats.bts.gov/ & https://www.ncei.noaa.gov/ | Contact: https://github.com/unimodular/Airplane_delay",
            style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #f8f9fa; z-index: 1000; text-align: center; padding: 10px;"
        )
    )
)



# Define server
def server(input, output, session):

    #--------------Flight Prediction---------------
    @reactive.Calc
    def input_validation_1():
        try:
            date = datetime(int(input.year1()), int(input.month1()), int(input.dayofmonth1()))
            if not (date.weekday()+1 == input.dayofweek1()):
                return f"The Day of Week should be {date.weekday()+1}."
        except ValueError as e:
            return "The selected date is not valid. Please check."
        
        future_7days = datetime.now() + timedelta(days=7)
        if date > future_7days or date < datetime.now():
            return "The selected date is not within 7 days from now. Please use the General Prediction."
        
        if not (0 <= input.dep_time1() <= 2359):
            return "The Scheduled Departure Time should be between 0 and 2359."
        if not (0 <= input.arr_time1() <= 2359):
            return "The Scheduled Departure Time should be between 0 and 2359."
        
        return None

    @output
    @render.text
    def error_message_1():
        return input_validation_1()

    def find_value(time, values):
        # time: UTC
        values = values['values']
        for value in values:
            time_part, duration_part = value['validTime'].split("/")
            start_time = parser.isoparse(time_part)
            match = re.search(r'\d+', duration_part[::-1]).group()
            duration_hours = int(match[::-1])
            duration = timedelta(hours=duration_hours)
            end_time = start_time + duration
            if time >= start_time and time < end_time:
                return value['value']
        return None
    
    def find_pressure_change(time, values):
        pressure_now = find_value(time, values)
        pressure_p1h = find_value(time - timedelta(hours=1), values)
        if pressure_now and pressure_p1h:
            return pressure_now-pressure_p1h
        return None
    
    def get_weather_data(origin, time):
        df = pd.read_csv('airports_lat_lon.csv',index_col='AIRPORT')
        lat, lon = df.loc[origin]
        grid_data = requests.get(f"https://api.weather.gov/points/{lat},{lon}").json()
        forecast_url = grid_data['properties']['forecastGridData']
        weather_response = requests.get(forecast_url).json()
        properties = weather_response['properties']
        
        # Extract weather data for model inputs
        data = [find_value(time, properties['dewpoint']),
                find_value(time, properties['temperature']),
                find_value(time, properties['quantitativePrecipitation']),
                find_pressure_change(time, properties['pressure']),
                find_value(time, properties['relativeHumidity']),
                find_value(time, properties['pressure']),
                find_value(time, properties['visibility']),
                find_value(time, properties['windSpeed'])
        ]
        
        weather_data = {
            'HourlyDewPointTemperature': 1.81601 if data[0] == None else data[0],
            'HourlyDryBulbTemperature': 8.90995 if data[1] == None else data[1],
            'HourlyPrecipitation': 0.0975287 if data[2] == None else data[2],
            'HourlyPressureChange': 0.033222 if data[3] == None else data[3],
            'HourlyRelativeHumidity': 65.2247 if data[4] == None else data[4],
            'HourlySeaLevelPressure': 1019.47 if data[5] == None else data[5], 
            'HourlyVisibility': 14.5233 if data[6] == None else data[6], 
            'HourlyWindSpeed': 3.67803 if data[7] == None else data[7]
        }
        return weather_data
    
    def prepare_input_data_1(user_input, weather_data):
        encoded_vars = [
            user_input.year1(), user_input.month1(), user_input.dayofmonth1(),
            user_input.dayofweek1(), user_input.marketing_airline1(), 
            user_input.origin1(), user_input.dest1()
        ]
        scaled_vars = [
            user_input.dep_time1(), user_input.arr_time1(),
            weather_data['HourlyDewPointTemperature'], weather_data['HourlyDryBulbTemperature'],
            weather_data['HourlyPrecipitation'], weather_data['HourlyPressureChange'],
            weather_data['HourlyRelativeHumidity'], weather_data['HourlySeaLevelPressure'],
            weather_data['HourlyVisibility'], weather_data['HourlyWindSpeed']
        ]
        return {'encoded': encoded_vars, 'scaled': scaled_vars}

    @reactive.event(input.predict_btn)
    def predict_delay_cancellation():
        # Check if input is valid
        if input_validation_1() != None:
            return

        # Get weather data
        hour = input.dep_time1() // 100
        minute = input.dep_time1() % 100
        timezone = pytz.timezone('America/Chicago')
        time = datetime(int(input.year1()), int(input.month1()), int(input.dayofmonth1()), hour, minute)
        time = timezone.localize(time)
        time = time.astimezone(pytz.UTC)
        weather_data = get_weather_data(input.origin1(), time)
        # Prepare input data for models
        model_data = prepare_input_data_1(input, weather_data)
        
        # Delay prediction
        delay_features = delay_scaler.transform([model_data['scaled']])
        delay_features = hstack([csr_matrix(delay_features), delay_encoder.transform([model_data['encoded']])])
        delay_prediction = delay_model.predict(delay_features)

        # Cancellation prediction
        cancel_features = cancel_scaler.transform([model_data['scaled']])
        cancel_features = hstack([csr_matrix(cancel_features), cancel_encoder.transform([model_data['encoded']])])
        cancel_prediction = cancel_model.predict_proba(cancel_features)[:, 1]

        return delay_prediction[0], cancel_prediction[0]

    @reactive.event(input.predict_btn)
    def get_api_message():
        # Check if input is valid
        if input_validation_1() != None:
            return

        # Get weather data
        hour = input.dep_time1() // 100
        minute = input.dep_time1() % 100
        timezone = pytz.timezone('America/Chicago')
        time = datetime(int(input.year1()), int(input.month1()), int(input.dayofmonth1()), hour, minute)
        time = timezone.localize(time)
        time = time.astimezone(pytz.UTC)
        weather_data = get_weather_data(input.origin1(), time)
        
        return weather_data

    @output
    @render.text
    def HourlyDewPointTemperature():
        return f"{round(get_api_message()['HourlyDewPointTemperature'],2)}"
    
    @output
    @render.text
    def HourlyDryBulbTemperature():
        return f"{round(get_api_message()['HourlyDryBulbTemperature'],2)}"
    
    @output
    @render.text
    def HourlyPrecipitation():
        return f"{round(get_api_message()['HourlyPrecipitation'],2)}"
    
    @output
    @render.text
    def HourlyPressureChange():
        return f"{round(get_api_message()['HourlyPressureChange'],2)}"
    
    @output
    @render.text
    def HourlyRelativeHumidity():
        return f"{round(get_api_message()['HourlyRelativeHumidity'],2)}"
    
    @output
    @render.text
    def HourlySeaLevelPressure():
        return f"{round(get_api_message()['HourlySeaLevelPressure'],2)}"
    
    @output
    @render.text
    def HourlyVisibility():
        return f"{round(get_api_message()['HourlyVisibility'],2)}"
    
    @output
    @render.text
    def HourlyWindSpeed():
        return f"{round(get_api_message()['HourlyWindSpeed'],2)}"
    
    @output
    @render.text
    def PredictedArrivalDelay():
        delay, _ = predict_delay_cancellation()
        return f"{delay} minutes"
    
    @output
    @render.text
    def PredictedCancellationProbability():
        _, cancel_prob = predict_delay_cancellation()
        return f"{cancel_prob*100}%"
    
    
    
    #--------------General Prediction---------------
    @reactive.Calc
    def input_validation_2():
        if input.year2() != "" and input.month2() != "" and input.dayofmonth2() != "":
            try:
                date = datetime(int(input.year2()), int(input.month2()), int(input.dayofmonth2()))
                if input.dayofweek2() != "":
                    if not (date.weekday()+1 == int(input.dayofweek2())):
                        return f"The Day of Week should be {date.weekday()+1}."
            except ValueError as e:
                return "The selected date is not valid. Please check."
        
        if input.dep_time2() != "":
            if not (0 <= int(input.dep_time2()) <= 2359):
                return "The Scheduled Departure Time should be between 0 and 2359."
        if input.arr_time2() != "":
            if not (0 <= int(input.arr_time2()) <= 2359):
                return "The Scheduled Arrival Time should be between 0 and 2359."
        
        return None
    
    @output
    @render.text
    def error_message_2():
        return input_validation_2()
    
    def prepare_input_data_2(user_input):
        if user_input.year2() != "":
            year2 = [int(user_input.year2())]
        else:
            year2 = [2018, 2019, 2020, 2021, 2022, 2023]
        
        if user_input.month2() != "":
            month2 = [int(user_input.month2())]
        else:
            month2 = [1,11,12]
        
        if user_input.dayofmonth2() != "":
            dayofmonth2 = [int(user_input.dayofmonth2())]
        else:
            dayofmonth2 = [i for i in range(1,32)]
            
        if user_input.dayofweek2() != "":
            dayofweek2 = [int(user_input.dayofweek2())]
        else:
            dayofweek2 = [1]
            
        if user_input.marketing_airline2() != "":
            marketing_airline2 = [user_input.marketing_airline2()]
        else:
            marketing_airline2 = all_airlines
            
        origin2 = [user_input.origin2()]
        
        dest2 = [user_input.dest2()]
            
        if user_input.dep_time2() != "":
            dep_time2 = [int(user_input.dep_time2())]
        else:
            dep_time2 = [1000]
            
        if user_input.arr_time2() != "":
            arr_time2 = [int(user_input.arr_time2())]
        else:
            arr_time2 = [1400]
            
        if user_input.hdpt() != "":
            hdpt = [float(user_input.hdpt())]
        else:
            hdpt = [1.81601]
            
        if user_input.hdbt() != "":
            hdbt = [float(user_input.hdbt())]
        else:
            hdbt = [8.90995]
            
        if user_input.hp() != "":
            hp = [float(user_input.hp())]
        else:
            hp = [0.0975287]
            
        if user_input.hpc() != "":
            hpc = [float(user_input.hpc())]
        else:
            hpc = [0.033222]
            
        if user_input.hrh() != "":
            hrh = [float(user_input.hrh())]
        else:
            hrh = [65.2247]
            
        if user_input.hsp() != "":
            hsp = [float(user_input.hsp())]
        else:
            hsp = [1019.47]
            
        if user_input.hv() != "":
            hv = [float(user_input.hv())]
        else:
            hv = [14.5233]
            
        if user_input.hws() != "":
            hws = [float(user_input.hws())]
        else:
            hws = [3.67803]
        
        combinations = list(itertools.product(year2, month2, dayofmonth2, dayofweek2,
                                              marketing_airline2, origin2, dest2,
                                              dep_time2, arr_time2,
                                              hdpt, hdbt, hp, hpc, hrh, hsp, hv, hws))
        
        df = pd.DataFrame(combinations, columns=['Year', 'Month', 'DayofMonth', 'DayOfWeek',
                                                 'Marketing_Airline_Network', 'Origin', 'Dest',
                                                 'CRSDepTime', 'CRSArrTime',
                                                 'HourlyDewPointTemperature', 'HourlyDryBulbTemperature',
                                                 'HourlyPrecipitation', 'HourlyPressureChange', 
                                                 'HourlyRelativeHumidity', 'HourlySeaLevelPressure',
                                                 'HourlyVisibility', 'HourlyWindSpeed'])

        month_days = {1: 31, 11: 30, 12: 31}
        df = df[df.apply(lambda row: row['DayofMonth'] <= month_days.get(row['Month']), axis=1)]
        df = df.rename(columns={'DayofMonth':'Day'})
        df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        df['DayOfWeek'] = df['date'].dt.dayofweek + 1
        df.drop(columns=['date'], inplace=True)
        df = df.rename(columns={'Day':'DayofMonth'})
        return df
    
    @reactive.event(input.general_predict_btn)
    def predict_general():
        # Check if input is valid
        if input_validation_2() != None:
            return
        
        model_data = prepare_input_data_2(input)
        encoder_cols = ['Year', 'Month', 'DayofMonth', 'DayOfWeek',
                        'Marketing_Airline_Network', 'Origin', 'Dest']
        scaler_cols = ['HourlyDewPointTemperature', 'HourlyDryBulbTemperature',
                       'HourlyPrecipitation', 'HourlyPressureChange',
                       'HourlyRelativeHumidity', 'HourlySeaLevelPressure',
                       'HourlyVisibility', 'HourlyWindSpeed', 'CRSDepTime', 'CRSArrTime']
        
        # Delay prediction for all values of missing optional fields
        delay_features = delay_scaler.transform(model_data[scaler_cols])
        delay_features = hstack([csr_matrix(delay_features), delay_encoder.transform(model_data[encoder_cols])])
        delay_prediction = delay_model.predict(delay_features)
        delay_mean, delay_min, delay_max = np.mean(delay_prediction), np.min(delay_prediction), np.max(delay_prediction)
        
        # Cancellation prediction for all values of missing optional fields
        cancel_features = cancel_scaler.transform(model_data[scaler_cols])
        cancel_features = hstack([csr_matrix(cancel_features), cancel_encoder.transform(model_data[encoder_cols])])
        cancel_prob = cancel_model.predict_proba(cancel_features)[:, 1]
        cancel_mean, cancel_min, cancel_max = np.mean(cancel_prob), np.min(cancel_prob), np.max(cancel_prob)

        return delay_mean, delay_min, delay_max, cancel_mean, cancel_min, cancel_max
    
    @output
    @render.text
    def AveragePredictedDelay():
        delay_mean, _, _, _, _, _ = predict_general()
        return f"Average Predicted Delay: {delay_mean:.2f} minutes"
    
    @output
    @render.text
    def MinimumPredictedDelay():
        _, delay_min, _, _, _, _ = predict_general()
        return f"Minimum Predicted Delay: {delay_min:.2f} minutes"
    
    @output
    @render.text
    def MaximumPredictedDelay():
        _, _, delay_max, _, _, _ = predict_general()
        return f"Maximum Predicted Delay: {delay_max:.2f} minutes"
    
    @output
    @render.text
    def AverageCancellationProbability():
        _, _, _, cancel_mean, _, _ = predict_general()
        return f"Average Cancellation Probability: {cancel_mean:.2%}"
    
    @output
    @render.text
    def MinimumCancellationProbability():
        _, _, _, _, cancel_min, _ = predict_general()
        return f"Minimum Cancellation Probability: {cancel_min:.2%}"
    
    @output
    @render.text
    def MaximumCancellationProbability():
        _, _, _, _, _, cancel_max = predict_general()
        return f"Maximum Cancellation Probability: {cancel_max:.2%}"
    
    @output
    @render.text
    def suggestions():
        delay_mean, _, _, cancel_mean, _, _ = predict_general()
        
        advice = []
        if delay_mean > 30:
            advice.append("Consider booking an earlier flight to avoid delays.")
        if cancel_mean > 0.2:
            advice.append("High cancellation risk. Be prepared for potential changes.")
        if not advice:
            advice.append("Your flight schedule looks optimal.")
        return "\n".join(advice)
    
        
    #--------------Statistical Data---------------
    @output
    @render_widget
    def airline_delay_plot():
        data = pd.read_csv("airline_stats.csv")
        selected_airlines = input.selected_airlines()
        if selected_airlines:
            if isinstance(selected_airlines, str):
                selected_airlines = [selected_airlines]
            data = data[data['Marketing_Airline_Network'].isin(selected_airlines)]
        data = data.sort_values(by='AvgDelay', ascending=False)
        fig = px.bar(data, 
                     x='Marketing_Airline_Network', 
                     y='AvgDelay', 
                     title="Average Delay by Airline Network",
                     labels={'AvgDelay': 'Average Delay (minutes)', 'Marketing_Airline_Network': 'Airline Network'})
        return fig

    
    @output
    @render_widget
    def airline_cancel_plot():
        data = pd.read_csv("airline_stats.csv")
        selected_airlines = input.selected_airlines()
        if selected_airlines:
            if isinstance(selected_airlines, str):
                selected_airlines = [selected_airlines]
            data = data[data['Marketing_Airline_Network'].isin(selected_airlines)]
        data = data.sort_values(by='CancelRate', ascending=False)
        fig = px.bar(data, 
                     x='Marketing_Airline_Network', 
                     y='CancelRate', 
                     title="Cancel Rate by Airline Network",
                     labels={'CancelRate': 'Cancel Rate', 'Marketing_Airline_Network': 'Airline Network'})
        return fig
    
    @output
    @render_widget
    def origin_delay_plot():
        data = pd.read_csv("origin_stats.csv")
        selected_origins = input.selected_origins()
        if selected_origins:
            if isinstance(selected_origins, str):
                selected_origins = [selected_origins]
            data = data[data['Origin'].isin(selected_origins)]
        data = data.sort_values(by='AvgDelay', ascending=False)
        fig = px.bar(data, 
                     x='Origin', 
                     y='AvgDelay', 
                     title="Average Delay by Origin Airport",
                     labels={'AvgDelay': 'Average Delay (minutes)', 'Origin': 'Origin Airport'})
        return fig

    
    @output
    @render_widget
    def origin_cancel_plot():
        data = pd.read_csv("origin_stats.csv")
        selected_origins = input.selected_origins()
        if selected_origins:
            if isinstance(selected_origins, str):
                selected_origins = [selected_origins]
            data = data[data['Origin'].isin(selected_origins)]
        data = data.sort_values(by='CancelRate', ascending=False)
        fig = px.bar(data, 
                     x='Origin', 
                     y='CancelRate', 
                     title="Cancel Rate by Origin Airport",
                     labels={'CancelRate': 'Cancel Rate', 'Origin': 'Origin Airport'})
        return fig
    
    @output
    @render_widget
    def dest_delay_plot():
        data = pd.read_csv("dest_stats.csv")
        selected_dests = input.selected_dests()
        if selected_dests:
            if isinstance(selected_dests, str):
                selected_dests = [selected_dests]
            data = data[data['Dest'].isin(selected_dests)]
        data = data.sort_values(by='AvgDelay', ascending=False)
        fig = px.bar(data, 
                     x='Dest', 
                     y='AvgDelay', 
                     title="Average Delay by Destination Airport",
                     labels={'AvgDelay': 'Average Delay (minutes)', 'Dest': 'Destination Airport'})
        return fig

    
    @output
    @render_widget
    def dest_cancel_plot():
        data = pd.read_csv("dest_stats.csv")
        selected_dests = input.selected_dests()
        if selected_dests:
            if isinstance(selected_dests, str):
                selected_dests = [selected_dests]
            data = data[data['Dest'].isin(selected_dests)]
        data = data.sort_values(by='CancelRate', ascending=False)
        fig = px.bar(data, 
                     x='Dest', 
                     y='CancelRate', 
                     title="Cancel Rate by Destination Airport",
                     labels={'CancelRate': 'Cancel Rate', 'Dest': 'Destination Airport'})
        return fig
    
app = App(app_ui, server)
