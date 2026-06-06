import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, classification_report, 
                              confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MultiLabelBinarizer, OneHotEncoder, LabelEncoder
import joblib

from utils import selected_features, selected_cat_cols, selected_num_cols, target_reg, target_cls


#  1. load data 
def load_data():
    df = pd.read_csv('data/processed/feature_engineered_df.csv')
    return df


#  2. load saved models and encoders 
def load_artifacts():
    reg_model   = joblib.load('models/best_model_reg.pkl')
    clf_model   = joblib.load('models/best_model_clf.pkl')
    scaler      = joblib.load('models/scaler_reg.pkl')
    ohe         = joblib.load('models/ohe_clf.pkl')
    mlb         = joblib.load('models/mlb_clf.pkl')
    le          = joblib.load('models/label_encoder.pkl')
    print("✅ All models and encoders loaded!")
    return reg_model, clf_model, scaler, ohe, mlb, le


#  3. prepare test data 
def prepare_test_data(df, ohe, mlb, scaler, le):

    X     = df[selected_features]
    Y_reg = df[target_reg]
    Y_cls = le.transform(df[target_cls])

    # split — must use same random_state as training
    
    # Regression split
    _, X_test_reg, _, Y_test_reg = train_test_split(
        X, Y_reg, test_size=0.3, random_state=42)

    # Classification split
    _, X_test_cls, _, Y_test_cls = train_test_split(
        X, Y_cls, test_size=0.3, random_state=42,
        stratify=Y_cls)
    
    
    # Regression test data
    x_reg_ohe = pd.DataFrame(
        ohe.transform(X_test_reg[['Campaign_Type', 'Brand']]),
        columns=ohe.get_feature_names_out(),
        index=X_test_reg.index
    )

    x_reg_channel = pd.DataFrame(
        mlb.transform(
            X_test_reg['Channel_Used'].str.split(', ')
        ),
        columns=mlb.classes_,
        index=X_test_reg.index
    )

    x_reg_num = X_test_reg[selected_num_cols]

    x_reg_scaled = pd.DataFrame(
        scaler.transform(x_reg_num),
        columns=x_reg_num.columns,
        index=x_reg_num.index
    )

    X_test_reg_final = pd.concat([ x_reg_scaled, x_reg_ohe, x_reg_channel ], axis=1)

    # Classification test data
    x_cls_ohe = pd.DataFrame(
        ohe.transform(X_test_cls[['Campaign_Type', 'Brand']]),
        columns=ohe.get_feature_names_out(),
        index=X_test_cls.index
    )

    x_cls_channel = pd.DataFrame(
        mlb.transform(
            X_test_cls['Channel_Used'].str.split(', ')
        ),
        columns=mlb.classes_,
        index=X_test_cls.index
    )

    x_cls_num = X_test_cls[selected_num_cols]

    x_cls_scaled = pd.DataFrame(
        scaler.transform(x_cls_num),
        columns=x_cls_num.columns,
        index=x_cls_num.index
    )

    X_test_cls_final = pd.concat([ x_cls_scaled, x_cls_ohe, x_cls_channel ],axis=1)

    print("✅ Test data prepared!")

    return (
        X_test_reg_final,
        Y_test_reg,
        X_test_cls_final,
        Y_test_cls
    )

   
# ── 4. evaluate regression ───────────────────────────────────
def evaluate_regression(reg_model, X_test_reg_final, Y_test_reg):

    y_pred = reg_model.predict(X_test_reg_final)
   

    r2   = r2_score(Y_test_reg, y_pred)
    mae  = mean_absolute_error(Y_test_reg, y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test_reg, y_pred))

    print("\n── Regression Evaluation ────────────────────────────")
    print(f"R2   : {r2:.4f}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")


    # plot actual vs predicted

    plt.figure(figsize=(8, 5))
    plt.scatter(Y_test_reg, y_pred, alpha=0.3, color='steelblue')
    plt.plot([Y_test_reg.min(), Y_test_reg.max()],
             [Y_test_reg.min(), Y_test_reg.max()],
             'r--', linewidth=2, label='Perfect prediction')
    plt.xlabel('Actual Revenue')
    plt.ylabel('Predicted Revenue')
    plt.title(f'Actual vs Predicted Revenue  |  R2: {r2:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/regression_actual_vs_pred.png')
    plt.show()
    print("✅ Regression plot saved!")

    return y_pred


# ── 5. evaluate classification ───────────────────────────────
def evaluate_classification(clf_model, X_test_cls_final, Y_test_cls, le):
    y_pred      = clf_model.predict(X_test_cls_final)
    y_pred_prob = clf_model.predict_proba(X_test_cls_final)[:, 1]

    acc     = accuracy_score(Y_test_cls, y_pred)
    roc_auc = roc_auc_score(Y_test_cls, y_pred_prob)

    print("\n── Classification Evaluation ────────────────────────")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print(f"\n{classification_report(Y_test_cls, y_pred, target_names=le.classes_)}")

    # confusion matrix plot
    cm = confusion_matrix(Y_test_cls, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix  |  Accuracy: {acc:.4f}')
    plt.tight_layout()
    plt.savefig('models/classification_confusion_matrix.png')
    plt.show()

    # ROC curve plot
    fpr, tpr, _ = roc_curve(Y_test_cls, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='steelblue', label=f'ROC AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'r--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — Profit/Loss Classification')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/classification_roc_curve.png')
    plt.show()
    print("✅ Classification plots saved!")


# ── main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    df                                        = load_data()
    reg_model, clf_model, scaler, ohe, mlb, le = load_artifacts()

    X_test_reg_final,Y_test_reg, X_test_cls_final, Y_test_cls = prepare_test_data(
                                                                                  df, ohe,mlb,scaler,le)
                                                
    evaluate_regression(reg_model,X_test_reg_final,Y_test_reg)
    evaluate_classification(clf_model, X_test_cls_final, Y_test_cls, le)

