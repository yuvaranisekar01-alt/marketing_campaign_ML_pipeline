import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_ingestion import insert_predictions, run_query
from src.utils import selected_num_cols

# ── page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Marketing Campaign Dashboard",
    page_icon  = "📊",
    layout     = "wide"
)

# ── load models and encoders ──────────────────────────────────
@st.cache_resource
def load_artifacts():
    reg_model = joblib.load('models/best_model_reg.pkl')
    clf_model = joblib.load('models/best_model_clf.pkl')
    reg_scaler = joblib.load('models/scaler_reg.pkl')
    clf_scaler = joblib.load('models/scaler_clf.pkl')
    ohe       = joblib.load('models/ohe_clf.pkl')
    mlb       = joblib.load('models/mlb_clf.pkl')
    le        = joblib.load('models/label_encoder.pkl')

    return reg_model, clf_model, reg_scaler, clf_scaler , ohe, mlb, le

reg_model, clf_model, reg_scaler, clf_scaler, ohe, mlb, le = load_artifacts()

# ── sidebar navigation ────────────────────────────────────────
st.sidebar.title("📊 Marketing Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔮 Prediction", "📈 Campaign Analytics", "🤖 Model Performance"]
)


# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("📊 Marketing Campaign Analysis")
    st.markdown("### Nykaa · Purplle · Tira — Campaign Intelligence Dashboard")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Brands", "3", "Nykaa, Purplle, Tira")
    with col2:
        st.metric("Classification Accuracy", "90%", "XGBoost")
    with col3:
        st.metric("Regression R²", "0.75", "RandomForest")
    with col4:
        st.metric("Total Features", "6", "After selection")

    st.markdown("---")
    st.markdown("""
    **What this dashboard does:**
    - 🔮 **Prediction** — Input campaign details, get Revenue and Profit/Loss predictions
    - 📈 **Campaign Analytics** — Explore brand, channel, and campaign performance
    - 🤖 **Model Performance** — View model metrics, confusion matrix, ROC curve
    """)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.title("🔮 Campaign Prediction")
    st.markdown("Enter campaign details to predict Revenue and Profit/Loss")
    st.markdown("---")

    # ── input form ────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Campaign Details")
        brand          = st.selectbox("Brand",         ["nykaa", "purplle", "tira"])
        campaign_type  = st.selectbox("Campaign Type", ["email", "influencer", "paid ads", "seo", "social media"])
        channels       = st.multiselect(
            "Channels Used",
            ["email", "facebook", "google", "instagram", "whatsapp", "youtube"],
            default=["instagram"]
        )

    with col2:
        st.subheader("Campaign Metrics")
        acquisition_cost  = st.number_input("Acquisition Cost (₹)",  min_value=0.0,  value=50000.0,  step=1000.0)
        conversion_rate   = st.number_input("Conversion Rate",        min_value=0.0,  max_value=1.0,  value=0.45,   step=0.01)
        cost_per_click    = st.number_input("Cost Per Click (₹)",     min_value=0.0,  value=25.0,     step=1.0)

    st.markdown("---")

    # ── predict button ────────────────────────────────────────
    if st.button("🔮 Predict", use_container_width=True):

        
        if not channels:
            st.error("Please select at least one channel!")
        else:
            # ── preprocess input ──────────────────────────────
            channel_str = ", ".join(sorted(channels))

            input_df = pd.DataFrame([{
                'Cost_per_click'  : cost_per_click,
                'Acquisition_Cost': acquisition_cost,
                'Conversion_rate' : conversion_rate,
                'Campaign_Type'   : campaign_type,
                'Brand'           : brand,
                'Channel_Used'    : channel_str }])

            # encode OHE
            ohe_cols = pd.DataFrame(
            ohe.transform(input_df[['Campaign_Type', 'Brand']]),
            columns=ohe.get_feature_names_out(),
            index=input_df.index)

            # encode MLB
            mlb_cols = pd.DataFrame(
            mlb.transform(input_df['Channel_Used'].str.split(', ')),
            columns=mlb.classes_ , index=input_df.index )
            
            num_input = input_df[['Cost_per_click', 'Acquisition_Cost', 'Conversion_rate']]

            # scale numeric
            num_scaled_reg = pd.DataFrame(
            reg_scaler.transform(num_input),
            columns=num_input.columns )

            num_scaled_clf = pd.DataFrame(
            clf_scaler.transform(num_input),
            columns=num_input.columns )
            

            # final input
            X_reg = pd.concat([num_scaled_reg, ohe_cols, mlb_cols],axis=1)

            X_clf = pd.concat([num_scaled_clf, ohe_cols, mlb_cols], axis=1)

            reg_cols = reg_model.feature_names_in_

            for col in reg_cols:
                if col not in X_reg.columns:
                    X_reg[col] = 0

            X_reg = X_reg[reg_cols]

            clf_cols = clf_model.get_booster().feature_names
            for col in clf_cols:
                if col not in X_clf.columns:
                    X_clf[col] = 0

            X_clf = X_clf[clf_cols]


            # predict
            pred_revenue    = reg_model.predict(X_reg)[0]
            pred_roi = ((pred_revenue - acquisition_cost)/ acquisition_cost)

            pred_profit_loss = ("Profit" if pred_roi > 0 else "Loss" )

            pred_clf = clf_model.predict(X_clf)[0]
            
            
            # ── results ───────────────────────────────────────
            st.markdown("### 📊 Prediction Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "💰 Predicted Revenue",
                    f"₹{pred_revenue:,.0f}"
                )
            if pred_roi > 0:
                st.success("🟢 Predicted Campaign Outcome: Profit")
            else:
                st.error("🔴 Predicted Campaign Outcome: Loss")
                

            with col3:
                net = pred_revenue - acquisition_cost
                st.metric(
                    "💹 Net (Revenue - Cost)",
                    f"₹{net:,.0f}",
                    delta=f"{'Positive' if net > 0 else 'Negative'}" )

            

            # ── input summary ─────────────────────────────────
            st.markdown("### 📋 Input Summary")
            st.dataframe(input_df, use_container_width=True)

            # ── save to DB ────────────────────────────────────
            try:
                pred_record = pd.DataFrame([{
                    'brand'                  : brand,
                    'campaign_type'          : campaign_type,
                    'channel_used'           : channel_str,
                    'acquisition_cost'       : acquisition_cost,
                    'predicted_revenue'      : pred_revenue,
                    'predicted_profit_loss'  : pred_profit_loss
                }])
                insert_predictions(pred_record)
                st.success("✅ Prediction saved to database!")
            except Exception as e:
                st.warning(f"Could not save to DB: {e}")


# ══════════════════════════════════════════════════════════════
# PAGE 3 — CAMPAIGN ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Campaign Analytics":
    st.title("📈 Campaign Analytics")
    st.markdown("---")

    # load data
    @st.cache_data
    def load_data():
        return pd.read_csv('data/processed/feature_engineered_df.csv')

    df = load_data()
    df.columns = df.columns.str.lower()

    # ── filters ───────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        brand_filter = st.multiselect(
            "Filter by Brand",
            df['brand'].unique().tolist(),
            default=df['brand'].unique().tolist()
        )
    with col2:
        type_filter = st.multiselect(
            "Filter by Campaign Type",
            df['campaign_type'].unique().tolist(),
            default=df['campaign_type'].unique().tolist()
        )

    df_filtered = df[
        (df['brand'].isin(brand_filter)) &
        (df['campaign_type'].isin(type_filter))
    ]

    st.markdown("---")

    # ── KPI metrics ───────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Campaigns", f"{len(df_filtered):,}")
    with col2:
        st.metric("Total Revenue",   f"₹{df_filtered['revenue'].sum():,.0f}")
    with col3:
        st.metric("Avg Revenue",     f"₹{df_filtered['revenue'].mean():,.0f}")
    with col4:
        profit_pct = (df_filtered['profit_loss'] == 'Profit').mean() * 100
        st.metric("Profit Rate",     f"{profit_pct:.1f}%")

    st.markdown("---")

    # ── charts ────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue by Brand")
        brand_rev = df_filtered.groupby('brand')['revenue'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=brand_rev, x='brand', y='revenue', ax=ax, palette='Blues_d')
        ax.set_title('Total Revenue by Brand')
        ax.set_xlabel('Brand')
        ax.set_ylabel('Revenue (₹)')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Profit vs Loss by Brand")
        pl_brand = df_filtered.groupby(['brand', 'profit_loss']).size().reset_index(name='count')
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=pl_brand, x='brand', y='count',
                    hue='profit_loss', palette={'Profit': '#2ecc71', 'Loss': '#e74c3c'}, ax=ax)
        ax.set_title('Profit vs Loss by Brand')
        st.pyplot(fig)
        plt.close()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue by Campaign Type")
        type_rev = df_filtered.groupby('campaign_type')['revenue'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=type_rev, x='campaign_type', y='revenue', ax=ax, palette='Greens_d')
        ax.set_title('Avg Revenue by Campaign Type')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Monthly Revenue Trend")
        monthly = df_filtered.groupby(['year', 'month'])['revenue'].sum().reset_index()
        monthly['period'] = monthly['year'].astype(str) + '-' + monthly['month'].astype(str).str.zfill(2)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(monthly['period'], monthly['revenue'], marker='o', color='steelblue')
        ax.set_title('Monthly Revenue Trend')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Regression — Revenue Prediction")
        st.metric("Test R²",  "0.75")
        st.metric("MAE",      "₹1,68,533")
        st.metric("RMSE",     "₹2,58,894")
        st.metric("Model",    "RandomForest Regressor")

        # load and show plot if exists
        if os.path.exists('models/regression_actual_vs_pred.png'):
            st.image('models/regression_actual_vs_pred.png',
                     caption='Actual vs Predicted Revenue')

    with col2:
        st.subheader("Classification — Profit/Loss")
        st.metric("Accuracy",  "90%")
        st.metric("Macro F1",  "0.85")
        st.metric("ROC AUC",   "0.85+")
        st.metric("Model",     "XGBoost Classifier")

        if os.path.exists('models/classification_confusion_matrix.png'):
            st.image('models/classification_confusion_matrix.png',
                     caption='Confusion Matrix')

    # ROC curve full width
    if os.path.exists('models/classification_roc_curve.png'):
        st.markdown("---")
        st.subheader("ROC Curve")
        st.image('models/classification_roc_curve.png',
                 caption='ROC Curve — Profit/Loss Classification')
        

        