import numpy as np
import pandas as pd
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,  MultiLabelBinarizer, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from xgboost import XGBRegressor

import joblib

from utils import target_reg, selected_features, selected_cat_cols, selected_num_cols

# 1.Load data 

def load_data():
    df = pd.read_csv('data/processed/feature_engineered_df.csv')
    print(f"✅ Data loaded! Shape: {df.shape}")
    return df

# 2. Prepare x and y

def prepare_data(df):
    X = df[selected_features]
    Y_reg = df[target_reg]
    print(f"✅ X shape: {X.shape} | Y shape: {Y_reg.shape}")
    return X, Y_reg

# 3. Split

def split_data(X, Y_reg):
    X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y_reg,
    test_size=0.3,
    random_state=42
    )
    print(f"✅ Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, Y_train, Y_test


# 4. Encode Features

def encode_features(X_train, X_test):

    ohe = OneHotEncoder(handle_unknown= 'ignore', sparse_output=False)

    x_train_ohe_cat = pd.DataFrame (
    ohe.fit_transform(X_train[['Campaign_Type', 'Brand']]),
    columns = ohe.get_feature_names_out(),
    index = X_train.index)

    X_test_ohe_cat = pd.DataFrame(
    ohe.transform(X_test[['Campaign_Type','Brand']]),
    columns=ohe.get_feature_names_out(),
    index=X_test.index)


    X_train_channel = X_train['Channel_Used'].str.split(', ')
    X_test_channel = X_test['Channel_Used'].str.split(', ')

    mlb = MultiLabelBinarizer()
    x_train_channel = pd.DataFrame(
        mlb.fit_transform(X_train['Channel_Used'].str.split(', ')),
        columns=mlb.classes_,
        index=X_train.index
    )
    x_test_channel = pd.DataFrame(
        mlb.transform(X_test['Channel_Used'].str.split(', ')),
        columns=mlb.classes_,
        index=X_test.index)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(ohe, 'models/ohe_reg.pkl')
    joblib.dump(mlb, 'models/mlb_reg.pkl')
    print("✅ Encoders saved!")
    
    return x_train_ohe_cat, X_test_ohe_cat, x_train_channel, x_test_channel

# 5. Scale numeric

def scale_features(X_train, X_test):

    x_train_num = X_train[selected_num_cols]
    x_test_num  = X_test[selected_num_cols]

    rs = RobustScaler()

    X_train_num_scaled = pd.DataFrame(
        rs.fit_transform(x_train_num),
        columns=x_train_num.columns,
        index=x_train_num.index)

    X_test_num_scaled = pd.DataFrame(
        rs.transform(x_test_num),
        columns= x_test_num.columns,
        index= x_test_num.index)
    
    joblib.dump(rs, 'models/scaler_reg.pkl')
    print("✅ Scaler saved!")

    return X_train_num_scaled, X_test_num_scaled

# 6. Build final feature matrix

def build_final(X_train_num_scaled, X_test_num_scaled,
                x_train_ohe_cat, X_test_ohe_cat,
                x_train_channel, x_test_channel):
    
    X_train_final = pd.concat([X_train_num_scaled,x_train_ohe_cat, x_train_channel ], axis = 1)
    X_test_final = pd.concat([X_test_num_scaled, X_test_ohe_cat, x_test_channel], axis=1)

    print(f"✅ Final train shape: {X_train_final.shape}")
    print(f"✅ Final test shape : {X_test_final.shape}")
    return X_train_final, X_test_final

# 7. train XG boost regressor

def train_model(X_train_final, Y_train):

    xgb_reg = XGBRegressor(n_estimators= 200, learning_rate= 0.03, max_depth = 5,
                     min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1)

    xgb = xgb_reg.fit(X_train_final, Y_train)

    joblib.dump(xgb, 'models/best_model_reg.pkl')
    print("✅ Regression model saved!")
    return xgb

 # 8. evaluate 

def evaluate_model(xgb_reg, X_train_final, X_test_final, Y_train, Y_test):

    y_train_pred = xgb_reg.predict(X_train_final)
    y_test_pred  = xgb_reg.predict(X_test_final)
    

    print("\n── Regression Results ───────────────────────────────")
    print(f"Train R2  : {r2_score(Y_train, y_train_pred):.4f}")
    print(f"Test  R2  : {r2_score(Y_test,  y_test_pred):.4f}")
    print(f"MAE       : {mean_absolute_error(Y_test, y_test_pred):.2f}")
    print(f"RMSE      : {np.sqrt(mean_squared_error(Y_test, y_test_pred)):.2f}")


 # main     
if __name__ == "__main__":

    df = load_data()
    X, Y_reg  = prepare_data(df)
    X_train, X_test, Y_train, Y_test = split_data(X, Y_reg)

    x_train_ohe, x_test_ohe, \
    x_train_channel, x_test_channel = encode_features(X_train, X_test)

    x_train_scaled, x_test_scaled  = scale_features(X_train, X_test)

    X_train_final, X_test_final = build_final(
                                            x_train_scaled, x_test_scaled,
                                            x_train_ohe,    x_test_ohe,
                                            x_train_channel, x_test_channel
                                       )
    xgb_reg = train_model(X_train_final, Y_train)
    
    evaluate_model(xgb_reg, X_train_final, X_test_final, Y_train, Y_test)
