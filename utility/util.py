import numpy as np 
import pandas as pd 
import plotly.graph_objects as go
from datetime import datetime, timedelta
import collections
import seaborn
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def create_training_dataset(first_reported_case_date, us_data, sliding_window=3):
    '''
    Creates the baseline training dataset that ends on March 31
    
    ..param first_reported_case_date: Date of first reported case
    ..paramtype first_reported_case_date: Datetime Object
    
    ..param us_df: Dataframe of US_data 
    ..paramtype usa_df: Pandas Dataframe
    
    ..param sliding_window: Controls how far back to look in time
        for each data point
    ..paramtype sliding_window: int
    
    ..return X: Training Dataframe
    ..rtype: Pandas Dataframe
    
    '''
    first_analysis_date = first_reported_case_date - timedelta(days=sliding_window)
    march_us_data = us_data[us_data['Date'] >= first_analysis_date].copy()
    dates = march_us_data.Date.unique()
    
    #Update Weather Stats for NA Values
    avg_states = {}
    for state in march_us_data['Province_State'].unique():
        state_df = march_us_data.query("Province_State == @state")
        temps = state_df[state_df['temperature'].isna() == False]['temperature']
        humidity = state_df[state_df['humidity'].isna() == False]['humidity']
        avg_temp = temps.mean()
        avg_humidity = humidity.mean()
        avg_states[state] = {'humidity': avg_humidity, 'temperature': avg_temp}
    march_us_data['temperature'] = march_us_data.apply(lambda x: avg_states[x['Province_State']]['temperature'] \
                                                       if np.isnan(x['temperature']) else x['temperature'], \
                                                       axis=1)
    march_us_data['humidity'] = march_us_data.apply(lambda x: avg_states[x['Province_State']]['humidity'] \
                                                       if np.isnan(x['humidity']) else x['humidity'], \
                                                       axis=1)
    
    last_day_analysis = datetime(2020, 3, 31)
    
    X = pd.DataFrame()
    analysis_dates = [(first_reported_case_date + timedelta(days=i)) \
                      for i in range(((last_day_analysis+timedelta(days=1))-first_reported_case_date).days)]

    for state in march_us_data['Province_State'].unique():
        df = march_us_data.query('Province_State == @state')
        for idx, date in enumerate(analysis_dates):
            single_point = {}
            for num in range(1,sliding_window+1):
                num_days_ago = date + timedelta(days=-num)
                num_days_df = df[df["Date"] == num_days_ago]
                cases = num_days_df['ConfirmedCases'].values[0]
                deaths = num_days_df['Fatalities'].values[0]
                single_point[f'Cases_Days_{num}'] = cases
                single_point[f"Deaths_Days_{num}"] = deaths

            today_df = df[df["Date"] == date]
    
            single_point['Date'] = date
            single_point['Days_First_Case'] = idx
            single_point['Province_State'] = state
            single_point['humidity'] = today_df['humidity'].values[0]
            single_point['temperature'] = today_df['temperature'].values[0]
            single_point['ConfirmedCases'] = today_df['ConfirmedCases'].values[0]
            single_point['Fatalities'] = today_df['Fatalities'].values[0]
            
    
            X = X.append(single_point,ignore_index=True)
        
    one_hot_states = pd.get_dummies(X['Province_State'])
    X = X.merge(one_hot_states, left_index=True, right_index=True)
        
    return X

def cumalitive_cases_plot(usa_df, state):
    '''
    Creates an interactive plot showing time series of cumalitive cases
    
    ..param usa_df: Dataframe of US_data (above)
    ..paramtype usa_df: Pandas Dataframe
    
    ..param state: state of analysis
    ..paramtype state: str
    
    ..return fig: Figure
    ..rtype: Plotly Figure
    '''
    
    if state not in usa_df['Province_State'].unique():
        return "State not in Dataframe"
    
    state_df = usa_df[usa_df["Province_State"] == state].copy()
    state_df.sort_values(by=['Month', "Day"], inplace=True)
    x = state_df['Date']
    y = state_df['ConfirmedCases']
    state_df['Text'] = state_df.apply(lambda x: "Temperature (C): " + str(x['temperature']) + \
                                     "<br>" + "Humidity: " + str(x['humidity']), axis=1)
    state_df['Marginal'] = state_df['ConfirmedCases'].diff()
    marginal = state_df['Marginal'].to_numpy()
    fatalities = state_df['Fatalities'].to_numpy()
    marginal_fatalities = state_df['Fatalities'].diff()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x,y=y, text=state_df['Text'], mode='markers+lines', marker=dict(size=6),name='Cumulative Cases'))
    fig.add_trace(go.Scatter(x=x,y=marginal, mode='markers+lines', name='Marginal Cases'))
    fig.add_trace(go.Scatter(x=x,y=fatalities, mode='markers+lines', name='Cumulative Fatalities'))
    fig.add_trace(go.Scatter(x=x,y=marginal_fatalities, mode='markers+lines', name='Marginal Fatalities'))
    
    fig.update_layout(title=dict(text=f'Confirmed Cases for State: {state}',y=0.9, x=0.5),
                      legend=dict(x=0.01,y=0.98), 
                      font=dict(size=12, color='black'),
                     xaxis_title='Date', yaxis_title='Cases',
                     yaxis_tickformat = '0f')

    
    return fig 

def create_top_5_states_plot(usa_df, top_five_states):
    '''
    Creates Plotly Plot for Top 5 US States
    
    ..param usa_df: Dataframe of US_data (above)
    ..paramtype usa_df: Pandas Dataframe
    
    ..param top_five_states: The top five states with COVID outbreak
    ..paramtype state: list
    
    ..return fig: Figure
    ..rtype: Plotly Figure
    '''
    fig = go.Figure()

    for state in top_five_states['Province_State'].unique():
        state_df = usa_df[usa_df["Province_State"] == state].copy()
        state_df.sort_values(by=['Month', "Day"], inplace=True)
        x = state_df['Date']
        y = state_df['ConfirmedCases']
        state_df['Text'] = state_df.apply(lambda x: "Temperature (C): " + str(x['temperature']) + \
                                         "<br>" + "Humidity: " + str(x['humidity']), axis=1)
        fig.add_trace(go.Scatter(x=x,y=y, text=state_df['Text'], mode='markers+lines',
                                 marker=dict(size=6), name=f'{state}'))

    fig.update_layout(title=dict(text=f'Confirmed Cases for Top 5 Worst States',y=0.9, x=0.5),
                          legend=dict(x=0.01,y=0.98), 
                          font=dict(size=12, color='black'),
                         xaxis_title='Date', yaxis_title='Cases',
                         yaxis_tickformat = '0f')
    
    return fig

def make_prediction_existing_data(train_X, train_Y, model, idx, states_dict, first_case_date, model_type='simple', verbose=True):
    '''
    Makes a prediction of existing data for analaysis
    
    ..param train_X: Training Feature Matrix
    ..paramtype train_X: numpy array
    
    ..param train_Y: Training Labels
    ..paramtype train_Y: numpy array
    
    ..param model: keras model used
    ..paramtype model: Keras Model (Sequential)
    
    ..param idx: Index to predict
    ..paramtype idx: int
    
    ..param states_dict: Dictionary that contains state and their index
        from the one hot encoding
    ..paramtype states_dict: dictionary
    
    ..param first_case_date: Date of first reported case
    ..paramtype first_case_date: Datetime Object
    
    ..param model_type: Defines simple or complex - used for prediction parsing
    ..paramtype model_type: str
    
    ..param verbose: Used to display results if True
    ..paramtype verbose: Bool
    
    ..return case_predictions: Predicted Confiremd Cumalitve Cases
    ..rtype: float
    
    ..return death_predictions: Predicted Fatalities
    ..rtype: float
    
    ..return prediction_date: Date of Prediction
    ..rtype: Datetime Object
    
    ..return prediction_state: State of Prediction
    ..rtype: str
    
    ..return actual_cases: Actual Number of Confirmed Cases
    ..rtype: int
    
    ..return actual_deaths: Actual Number of Fatalities
    ..rtype: int
    '''
    predict_X = train_X[idx-1: idx,:]
    predict_Y = train_Y[idx-1: idx,:]
    predictions = model.predict(predict_X)
    prediction_states = predict_X[0][9:]
    for idx in range(len(prediction_states)):
        if prediction_states[idx] == 1:
            good_index = idx
    prediction_state = states_dict[good_index]

    prediction_date = first_case_date + timedelta(days=predict_X[0][4])
    actual_cases = predict_Y[0][0]
    actual_deaths = predict_Y[0][1]
    case_predictions = predictions[0][0] if model_type == 'simple' else predictions [0][0][0]
    death_predictions = predictions[0][1] if model_type == 'simple' else predictions [1][0][0]
    if verbose:
        print(f"Date of Predicition: {prediction_date}")
        print(f"State of Prediction: {prediction_state}")
        print("\n")
        print(f"Confirmed Cases Prediction: {case_predictions:.2f}")
        print(f"Actual Confirmed Cases: {predict_Y[0][0]}")
        print("\n")
        print(f"Confirmed Fatalities Prediction: {death_predictions:.2f}")
        print(f"Actual Confirmed Fatalities: {predict_Y[0][1]}")
    return case_predictions, death_predictions, prediction_state, prediction_date, actual_cases, actual_deaths


def compare_models(train_X, train_Y, models, idx, states_dict, first_case_date, verbose=True):
    '''
    Compare models based on how far they are from actuals
    
    ..param train_X: Training Feature Matrix
    ..paramtype train_X: numpy array
    
    ..param train_Y: Training Labels
    ..paramtype train_Y: numpy array
    
    ..param models: List that contains the Keras Models
    ..paramtype models: list
    
    ..param idx: Index to predict
    ..paramtype idx: int
    
    ..param states_dict: Dictionary that contains state and their index
        from the one hot encoding
    ..paramtype states_dict: dictionary
    
    ..param first_case_date: Date of first reported case
    ..paramtype first_case_date: Datetime Object
    
    ..param verbose: Used to display results if True
    ..paramtype verbose: Bool
    
    ..return d: Dictionary that contains the model predictions
    ..rtype: dictionary
    
    ..return best_case_pred: Best prediction for cumalitive cases from both models
    ..rtype: float
    
    ..return best_death_pred: Best prediction for fatalities from both models
    ..rtype: float
    '''
    
    model_types = {0: 'simple', 1: 'complex'}
    d = {}
    
    for i,model in enumerate(models):
        case_predictions, death_predictions, prediction_state, prediction_date, actual_cases, actual_deaths = \
            make_prediction_existing_data(train_X, train_Y, model, idx, states_dict, first_case_date,
                                          model_type=model_types[i], verbose=False)        
        if i == 0 and verbose:
#             confirmed_cases_prediction = preds[0][0]
#             confirmed_deaths_prediction = preds[0][1]
            print(f"Prediction State: {prediction_state}")
            print(f"Prediction Date: {prediction_date}")
            print(f"Actual Cases: {actual_cases}")
            print(f"Actual Deaths: {actual_deaths}")
            print("\n")
                
        confirmed_cases_prediction = case_predictions
        confirmed_deaths_prediction = death_predictions
        case_difference = abs(confirmed_cases_prediction - actual_cases)
        death_difference = abs(confirmed_deaths_prediction - actual_deaths)
        d[i] = {"Cases": confirmed_cases_prediction, "Deaths": confirmed_deaths_prediction,
                "Case Difference": case_difference, "Death Difference": death_difference}
        if verbose:
            print(f"Model: {i+1}")
            print(f"Predicted Cases: {confirmed_cases_prediction}")
            print(f"Predicted Deaths: {confirmed_deaths_prediction}")
            print("\n")
        
    case_diff_min = None
    death_diff_min = None
    best_case_model = None
    best_death_model = None
    best_case_pred = None
    best_death_pred = None
    
    if d[0]['Case Difference'] > d[1]['Case Difference']:
        best_case_model = 1
        case_diff_min = d[1]['Case Difference']
        best_case_pred = d[1]['Cases']
    else:
        best_case_model = 0
        case_diff_min = d[0]['Case Difference']
        best_case_pred = d[0]['Cases']
        
    if d[0]['Death Difference'] > d[1]['Death Difference']:
        best_death_model = 1
        death_diff_min = d[1]['Death Difference']
        best_death_pred = d[1]['Deaths']
    else:
        best_death_model = 0
        death_diff_min = d[0]['Death Difference']
        best_death_pred = d[0]['Deaths']
    
    if verbose:
        print(f"Best Case Prediction Model: Model {best_case_model} with a difference of {case_diff_min*1:0.2f}")
        print(f"Best Fatality Prediction Model: Model {best_death_model} with a difference of {death_diff_min*1:0.2f}")
    
        
    return d, best_case_pred, best_death_pred

def create_state_prediction_df(train_X, train_Y, shuffled_X, state, models, states_dict, first_case_date):
    '''
    Creates a dataframe with the predictions from the models for a state
    
    ..param train_X: Training Feature Matrix
    ..paramtype train_X: numpy array
    
    ..param train_Y: Training Labels
    ..paramtype train_Y: numpy array
    
    ..param shuffled_X: Training Dataframe
    ..paramtype shuffled_X: Pandas Dataframe
    
    ..param state: State of analysis
    ..paramtype state: str
    
    ..param models: List that contains the Keras Models
    ..paramtype models: list
    
    ..param states_dict: Dictionary that contains state and their index
        from the one hot encoding
    ..paramtype states_dict: dictionary
    
    ..param first_case_date: Date of first reported case
    ..paramtype first_case_date: Datetime Object
    
    ..return df: Dataframe with predictions by date
    ..rtype: Pandas Dataframe
    '''
    
    state_indexes = list(shuffled_X[shuffled_X['Province_State'] == state].sort_values("Date", ascending=True).index.values)
    state_indexes = [ele+1 for ele in state_indexes]
    dates = list(shuffled_X[shuffled_X['Province_State'] == state].sort_values("Date", ascending=True)["Date"])
    cases = []
    deaths = []
    for i in state_indexes:
        d, case_pred, death_pred = compare_models(train_X, train_Y, models, i, states_dict, first_case_date, verbose=False)
        cases.append(case_pred)
        deaths.append(death_pred)

    state_dict = {"Date": dates, 'Predicted_Cases': cases, "Predicted_Deaths": deaths}
    df = pd.DataFrame(state_dict)
    df['Province_State'] = state
    
    return df

def plot_prediction_actual_comparision(prediction_df, us_data, state, first_case_date):
    '''
    Creates a Plot that shows difference in prediction and actuals
    
    ..param prediction_df: Dataframe with predictions for a state
    ..paramtype prediction_df: Pandas Dataframe
    
    ..param us_data: Dataframe with US data
    ..paramtype us_data: Pandas Dataframe
    
    ..param state: State of Analysis
    ..paramtype state: str
    
    ..param first_case_date: Date of first reported case
    ..paramtype first_case_date: Datetime Object
    
    ..return fig: Figure
    ..rtype: Plotly Figure
    '''

    x = prediction_df['Date']
    y = prediction_df['Predicted_Cases']
    fatalities = prediction_df['Predicted_Deaths'].to_numpy()

    state_filter = us_data['Province_State'] == state
    date_filter = us_data['Date'] >= first_case_date
    us_data[(state_filter) & date_filter]
    filtered_df = us_data[(state_filter) & date_filter].sort_values('Date')[['ConfirmedCases', "Fatalities"]]
    cases_true = filtered_df['ConfirmedCases'].to_numpy()
    deaths_true = filtered_df['Fatalities'].to_numpy()

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode='markers+lines',
            marker=dict(size=6),
            name='Predicted Cases',
            line=dict(width=4, dash='dash')
        )
    )
    fig.add_trace(go.Scatter(x=x,y=cases_true, mode='markers+lines', name='Actual Cases',
                            line=dict(width=4)))
    fig.add_trace(go.Scatter(x=x,y=fatalities, mode='markers+lines', name='Predicted Fatalities',
                            line=dict(width=4, dash='dash')))
    fig.add_trace(go.Scatter(x=x,y=deaths_true, mode='markers+lines', name='Actual Fatalities',
                            line=dict(width=4)))

    fig.update_layout(title=dict(text=f'Confirmed and Predicted Cases for State: {state}',y=0.9, x=0.5),
                      legend=dict(x=0.01,y=0.98), 
                      font=dict(size=12, color='black'),
                     xaxis_title='Date', yaxis_title='Cases',
                     yaxis_tickformat = '0f')
    return fig


def create_training_val_loss_plot(model):
    print(f"Minimum Training Loss (Square Rooted): {min(model.history.history['loss'])**0.5}")
    print(f"Occured at Epoch: {np.argmin(model.history.history['loss'])}")
    print(f"Minimum Validation Loss (Square Rooted): {min(model.history.history['val_loss'])**0.5}")
    print(f"Occured at Epoch: {np.argmin(model.history.history['val_loss'])}")


    plt.rcParams.update({'text.color' : "white",
                         'axes.labelcolor' : "white",
                        'xtick.color': 'white',
                        'ytick.color': 'white'})

    plt.title("Square Root Losses")
    plt.plot(np.sqrt(model.history.history['loss']), 'b', \
            np.sqrt(model.history.history['val_loss']),'r',)
    legend = plt.legend(labels=['Training Loss', 'Validation Loss'])
    plt.setp(legend.get_texts(), color='black')

    plt.show()
    
def create_predictions_up_to(previous_state_predictions, first_reported_case_date, last_date, state, shuffled_X,
                             sliding_window, good_col_order, states_dict,
                             model, model_type):
    '''
    Creates Predictions from March 31 to specific date
    
    ..param previous_state_predictions: All of the previous predictions in March
    ..paramtype previous_state_predictions: Pandas Dataframe
    
    ..param first_reported_case_date: Date of first reported case
    ..paramtype first_reported_case_date: Datetime Object
    
    ..param last_date: Date of last analysis, i.e. today
    ..paramtype last_date: Datetime Object
    
    ..param state: State of analysis
    ..paramtype state: str
    
    ..param shuffled_X: Training Dataframe
    ..paramtype shuffled_X: Pandas Dataframe
    
    ..param sliding_window: Controls how far back to look in time
        for each data point
    ..paramtype sliding_window: int
    
    ..param good_col_order: Order of columns necessary for good prediction
    ..paramtype good_col_order: list
    
    ..param states_dict: Dictionary that contains state and their index
        from the one hot encoding
    ..paramtype states_dict: dictionary
    
    ..param model: keras model used
    ..paramtype model: Keras Model (Sequential)
    
    ..param model_type: Model Type
    ..paramtype model_type: str
    
    ..return all_predictions_to_date: Dataframe with predictions up to date
    ..rtype: Pandas Dataframe
    '''
    
    march_31 = datetime(2020, 3, 31)
    days_since_march_31 = last_date - march_31
    new_training_data = shuffled_X[shuffled_X['Province_State'] == state]
    last_humidity = new_training_data.sort_values("Date", ascending=False)['humidity'].values[0]
    last_temp = new_training_data.sort_values("Date", ascending=False)['temperature'].values[0]
    return_df = pd.DataFrame()
    for day in range(1,days_since_march_31.days+1):
        date = march_31 + timedelta(days=day)
        new_data_point = {}
        new_data_point['humidity'] = last_humidity
        new_data_point['temperature'] = last_temp
        new_data_point['Days_First_Case'] = (date-first_reported_case_date).days
        for num in range(1,sliding_window+1):
            num_days_ago = date + timedelta(days=-num)
            num_days_df = new_training_data[new_training_data["Date"] == num_days_ago]
            cases = num_days_df['ConfirmedCases'].values[0]
            deaths = num_days_df['Fatalities'].values[0]
            new_data_point[f'Cases_Days_{num}'] = cases
            new_data_point[f"Deaths_Days_{num}"] = deaths
        for state_val in states_dict.values():
            new_data_point[state_val] = 0 if state_val != state else 1
        
        new_data_df = pd.DataFrame(new_data_point, index=[0])
        new_data_numpy = new_data_df[good_col_order].to_numpy()
        predictions = model.predict(new_data_numpy)
        if model_type == 'simple':
            predict_cases = predictions[0][0]
            predict_deaths = predictions[0][1]
        else:
            predict_cases = predictions[0][0][0]
            predict_deaths = predictions[1][0][0]
            
        #Needed just for construction purposes
        new_data_df['Old Index'] = 5 #not relevant
        new_data_df['Date'] = date
        new_data_df['ConfirmedCases'] = predict_cases
        new_data_df['Fatalities'] = predict_deaths
        new_data_df['Province_State'] = state
        new_training_data = new_training_data.append(new_data_df, sort=True)
        _df = pd.DataFrame({"Date": date, "Province_State": state, "Predicted_Cases": predict_cases,
                           "Predicted_Deaths": predict_deaths}, index=[0])
        return_df = return_df.append(_df)
    all_predictions_to_date = previous_state_predictions.append(return_df, sort=True)
    
    return all_predictions_to_date

def create_new_training_point(shuffled_X, state, date, cases, fatalities,
                              first_reported_case_date,sliding_window, good_col_order,
                              states_dict,humidity=None, temp=None):
    '''
    Creates a new training point with additional data
    
    ..param shuffled_X: Original March dataset
    ..paramtype shuffled_X: Pandas Dataframe
    
    ..param state: State of analysis
    ..paramtype state: str
    
    ..param date: Date being added
    ..paramtype date: Datetime object
    
    ..param cases: Confirmed Cumaltive Cases added
    ..paramtype cases: int
    
    ..param fatalities: Confirmed Fatalities added
    ..paramtype cases: int
    
    ..param first_reported_case_date: Date of first reported case
    ..paramtype first_reported_case_date: Datetime Object
    
    ...param sliding_window: Controls how far back to look in time
        for each data point
    ..paramtype sliding_window: int
    
    ..param good_col_order: Order of columns necessary for good prediction
    ..paramtype good_col_order: list
    
    ..param states_dict: Dictionary that contains state and their index
        from the one hot encoding
    ..paramtype states_dict: dictionary
    '''
    new_training_data = shuffled_X[shuffled_X['Province_State'] == state]
    last_humidity = new_training_data.sort_values("Date", ascending=False)['humidity'].values[0]
    last_temp = new_training_data.sort_values("Date", ascending=False)['temperature'].values[0]
    new_data_point = {}
    new_data_point['humidity'] = last_humidity if not humidity else humidity
    new_data_point['temperature'] = last_temp if not temp else temp
    new_data_point['Days_First_Case'] = (date-first_reported_case_date).days
    for num in range(1,sliding_window+1):
        num_days_ago = date + timedelta(days=-num)
        num_days_df = new_training_data[new_training_data["Date"] == num_days_ago]
        cases = num_days_df['ConfirmedCases'].values[0]
        deaths = num_days_df['Fatalities'].values[0]
        new_data_point[f'Cases_Days_{num}'] = cases
        new_data_point[f"Deaths_Days_{num}"] = deaths
    for state_val in states_dict.values():
        new_data_point[state_val] = 0 if state_val != state else 1
        
    new_data_df = pd.DataFrame(new_data_point, index=[0])
    new_features_point = new_data_df[good_col_order].to_numpy()
    
    new_labels_point = np.array([cases, fatalities])
    
    return new_features_point, new_labels_point


#DEPRECATED
def create_april_1_prediction(first_reported_case_date, state, shuffled_X, sliding_window, good_col_order):
    april_1 =  datetime(2020, 4, 1)
    days_since_first = april_1 - first_reported_case_date
    new_training_data = shuffled_X[shuffled_X['Province_State'] == state]
    new_data_point = {}
    new_data_point['humidity'] = new_training_data.sort_values("Date", ascending=False)['humidity'].values[0]
    new_data_point['temperature'] = new_training_data.sort_values("Date", ascending=False)['temperature'].values[0]
    for num in range(1,sliding_window+1):
        num_days_ago = april_1 + timedelta(days=-num)
        num_days_df = new_training_data[new_training_data["Date"] == num_days_ago]
        cases = num_days_df['ConfirmedCases'].values[0]
        deaths = num_days_df['Fatalities'].values[0]
        new_data_point[f'Cases_Days_{num}'] = cases
        new_data_point[f"Deaths_Days_{num}"] = deaths

    new_data_point['Days_First_Case'] = days_since_first.days
    
    for state_val in states_dict.values():
        new_data_point[state_val] = 0 if state_val != state else 1

    new_data_df = pd.DataFrame(new_data_point, index=[0])
    return new_data_df[good_col_order].to_numpy(), new_data_df[good_col_order]