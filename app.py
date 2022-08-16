import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import json
from flask import Flask, request, jsonify
from datetime import date
import datetime as dt


app = Flask(__name__)

#@app.route("/")

#def index():
#    return jsonify("hello World")

    
@app.route('/', methods = ['GET','POST'])


def predict():
    df = pd.read_excel("MG Data for insight.xlsx", sheet_name = 'Sheet1')
    df["Posting Date"] = pd.to_datetime(df["Posting Date"])
    dt_ticket_count = df.groupby(["Posting Date"]).agg({"Status" : "count"})
    Description = df.groupby('Posting Date')['Description'].value_counts().unstack().fillna(0).astype(int)
    Zone = df.groupby('Posting Date')['Zone'].value_counts().unstack().fillna(0).astype(int)
    TransactionType = df.groupby('Posting Date')['TRANSACTIONTYPE'].value_counts().unstack().fillna(0).astype(int)
    Refined_data = pd.concat([Description, dt_ticket_count], axis = 1, join = "inner")
    Distributed_data = Refined_data.reset_index()
    d_ticket_count = dt_ticket_count.drop(dt_ticket_count.index[4::7])
    _ticket_count = d_ticket_count.reset_index(inplace = False)
    def create_features(df, label=None):
        df['date'] = df.index
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.weekofyear
    
        X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
        if label:
            y = df[label]
            return X, y
        return X
    X, y = create_features(dt_ticket_count, label='Status')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 20)
    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
           verbose=False)
    y_pred = reg.predict(X_test) # 1st model completed
    y_pred = pd.Series(reg.predict(X_test), index=X_test.index)
    distributed_x = Refined_data.iloc[:,::-1]
    X_tr, X_test_distributed, y_tr, y_te = train_test_split(distributed_x, y, test_size = 0.25, random_state = 20)
    X_test_distributed.drop("Status", axis = 1, inplace = True)
    data =pd.concat({"True_values": y_test, "Predicted_values": y_pred.apply(np.ceil).astype(int)}, axis = 1)  

    Test_data = pd.concat([data,X_test_distributed], axis = 1)
    Test_data["Difference"] = abs(Test_data["True_values"] - Test_data["Predicted_values"])
    MultiOutputReg_data = Refined_data

    X,y = MultiOutputReg_data.iloc[:,-1],MultiOutputReg_data.iloc[:,:-1]
    y.drop(["Change Request","Service Request"], inplace = True, axis = 1)
    X_train_catg, X_test_catg, y_train_catg, y_test_catg = train_test_split(X, y, test_size = 0.25, random_state = 20)
    regr = MultiOutputRegressor(Ridge(random_state=123)).fit(np.array(X_train_catg).reshape(595,1), y_train_catg)
    y_pred_catg = regr.predict(np.array(X_test_catg).reshape(199,1))# 2nd predict model
    y_pred = pd.DataFrame(y_pred_catg)
    Categories = df[df['Description'] == "High"]
    Categories2 = df[df['Description'] == "Very High"]
    Categories = pd.concat([Categories, Categories2])
    Categories["Posting Date"] = pd.to_datetime(df["Posting Date"])
    Categories = Categories.groupby('Posting Date')['CATEGORY'].value_counts().unstack().fillna(0).astype(int)
    Categories["Total_tickets"] = Categories.iloc[:,:-1].sum(axis = 1)
    X,y = Categories.iloc[:,-1],Categories.iloc[:,:-1]
    X_train_catg, X_test_catg, y_train_catg, y_test_catg = train_test_split(X, y, test_size = 0.25, random_state = 20)
    regr = MultiOutputRegressor(Ridge(random_state=123)).fit(np.array(X_train_catg).reshape(581,1), y_train_catg)
    y_pred_catg = regr.predict(np.array(X_test_catg).reshape(194,1))
    y_pred = pd.DataFrame(y_pred_catg)
    x_future_date = pd.date_range(start ="2022-08-01", end = "2023-01-31")

    x_future_dates = pd.DataFrame()

    x_future_dates["Dates"] = pd.to_datetime(x_future_date)

    x_future_dates.index = x_future_dates["Dates"]
    X, y = create_features(x_future_dates, label='Dates')


    y_future_total_tickets = reg.predict(X)
    x_future_dates["Predicted Tickets"] = y_future_total_tickets
    x_future_dates.drop("Dates", inplace = True, axis = 1)
    y_future_prediction = regr.predict(np.array(x_future_dates["Predicted Tickets"]).reshape(184,1))
    y_future_prediction = pd.DataFrame(y_future_prediction)
    y_future_prediction.rename(columns = {0:'After Sales',
                                      1:'C4C',
                                      2:'Dashboard',
                                      3:'Finance',
                                      4:'HCM',
                                      5:'Hybris Marketing',
                                      6:'Others',
                                      7:'Parts',
                                      8:'Sales',
                                      9:'Success Factors',
                                      10:'Warranty'}, inplace = True)

    y_future_prediction.index = x_future_dates.index
    future_tickets_prediction =pd.concat([x_future_dates["Predicted Tickets"], y_future_prediction],axis = 1)
    future_tickets_prediction.astype(int)
    ans = future_tickets_prediction.astype(int)
    result = ans.to_json(orient="table")
    parsed = json.loads(result)

    return jsonify({"parsed":parsed})


if __name__ == "__main__":
        app.run()






