import numpy as np
import pandas as pd
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MultiLabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib

from utils import selected_features, selected_cat_cols, selected_num_cols, target_cls


# 1. load data 

def load_data():
    df = pd.read_csv('data/processed/feature_engineered_df.csv')
    print(f"✅ Data loaded! Shape: {df.shape}")
    return df


# 2. prepare X and y

def prepare_data(df):
    X     = df[selected_features]
    Y_cls = df[target_cls]
    print(f"✅ X shape: {X.shape} | Y shape: {Y_cls.shape}")
    return X, Y_cls


# 3. encode target variable for XGBoost Classifier

def encode_target(Y_cls):
    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y_cls)   # Loss=0, Profit=1
    joblib.dump(le, 'models/label_encoder.pkl')
    print(f"✅ Target encoded! Classes: {le.classes_}")
    return Y_encoded, le


#  4. split 

def split_data(X, Y_encoded):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=Y_encoded        # stratify for classification
    )
    print(f"✅ Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, Y_train, Y_test


# 5. encode features 

def encode_features(X_train, X_test):

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    x_train_ohe = pd.DataFrame(
        ohe.fit_transform(X_train[['Campaign_Type', 'Brand']]),
        columns=ohe.get_feature_names_out(),
        index=X_train.index
    )
    x_test_ohe = pd.DataFrame(
        ohe.transform(X_test[['Campaign_Type', 'Brand']]),
        columns=ohe.get_feature_names_out(),
        index=X_test.index
    )

    mlb = MultiLabelBinarizer()
    x_train_channel = pd.DataFrame(
        mlb.fit_transform(X_train['Channel_Used'].str.split(', ')),
        columns=mlb.classes_,
        index=X_train.index
    )
    x_test_channel = pd.DataFrame(
        mlb.transform(X_test['Channel_Used'].str.split(', ')),
        columns=mlb.classes_,
        index=X_test.index
    )

    joblib.dump(ohe, 'models/ohe_clf.pkl')
    joblib.dump(mlb, 'models/mlb_clf.pkl')
    print("✅ Encoders saved!")

    return x_train_ohe, x_test_ohe, x_train_channel, x_test_channel


# 6. scale numeric 

def scale_features(X_train, X_test):
    x_train_num = X_train[selected_num_cols]
    x_test_num  = X_test[selected_num_cols]

    rs = RobustScaler()
    x_train_scaled = pd.DataFrame(
        rs.fit_transform(x_train_num),
        columns=x_train_num.columns,
        index= x_train_num.index
    )
    x_test_scaled = pd.DataFrame(
        rs.transform(x_test_num),
        columns= x_test_num.columns,
        index= x_test_num.index
    )

    joblib.dump(rs, 'models/scaler_clf.pkl')
    print("✅ Scaler saved!")

    return x_train_scaled, x_test_scaled


# 7. build final feature matrix

def build_final(x_train_scaled, x_test_scaled,
                x_train_ohe, x_test_ohe,
                x_train_channel, x_test_channel):

    X_train_final = pd.concat(
        [x_train_scaled, x_train_ohe, x_train_channel], axis=1
    )
    X_test_final = pd.concat(
        [x_test_scaled, x_test_ohe, x_test_channel], axis=1
    )

    print(f"✅ Final train shape: {X_train_final.shape}")
    print(f"✅ Final test shape : {X_test_final.shape}")
    return X_train_final, X_test_final


#  8. train XGBoost classifier 

def train_model(X_train_final, Y_train):
    xgb_cls = XGBClassifier( random_state=42,)
    

    xgb_cls.fit(X_train_final, Y_train)

    joblib.dump(xgb_cls, 'models/best_model_clf.pkl')
    print("✅ Classification model saved!")
    return xgb_cls


# 9. evaluate 

def evaluate_model(xgb_cls, X_train_final, X_test_final,
                   Y_train, Y_test, le):
    
    y_test_pred  = xgb_cls.predict(X_test_final)
    y_train_pred = xgb_cls.predict(X_train_final)

    print("\n── Classification Results ───────────────────────────")
    print(f"Train Accuracy : {accuracy_score(Y_train, y_train_pred):.4f}")
    print(f"Test  Accuracy : {accuracy_score(Y_test,  y_test_pred):.4f}")
    print(f"\n{classification_report(Y_test, y_test_pred, target_names=le.classes_)}")
    print(f"Confusion Matrix:\n{confusion_matrix(Y_test, y_test_pred)}")


# ── main ─────────────────────────────────────────────────────
if __name__ == "__main__":

    df  = load_data()

    X, Y_cls  = prepare_data(df)
    Y_encoded, le  = encode_target(Y_cls)
    X_train, X_test, Y_train, Y_test = split_data(X, Y_encoded)

    x_train_ohe, x_test_ohe, \
    x_train_channel, x_test_channel = encode_features(X_train, X_test)

    x_train_scaled, x_test_scaled  = scale_features(X_train, X_test)

    X_train_final, X_test_final  = build_final(
                                            x_train_scaled, x_test_scaled,
                                            x_train_ohe,    x_test_ohe,
                                            x_train_channel, x_test_channel )
    
    xgb_cls  = train_model(X_train_final, Y_train)

    evaluate_model(xgb_cls, X_train_final, X_test_final, Y_train, Y_test, le)

    