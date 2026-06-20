import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("models/car_price_model.pkl")
    selected_features = joblib.load("models/selected_features.pkl")
    return model, selected_features

@st.cache_data
def load_results():
    with open("outputs/results.json") as f:
        return json.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("data/car data.csv")
    return df

@st.cache_resource
def load_vision_model():
    return MobileNetV2(weights='imagenet')

CAR_KEYWORDS = ['car', 'convertible', 'jeep', 'limousine', 'minivan', 'pickup',
                'sports_car', 'beach_wagon', 'racer', 'station_wagon',
                'model_t', 'amphibian']

def is_car_match(label):
    label = label.lower()
    return any(
        kw == label or label.startswith(kw + '_') or label.endswith('_' + kw)
        for kw in CAR_KEYWORDS
    )

def check_is_car(uploaded_image):
    vision_model = load_vision_model()
    img = Image.open(uploaded_image).convert('RGB').resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = vision_model.predict(img_array, verbose=0)
    decoded = decode_predictions(preds, top=5)[0]

    is_car = any(
        is_car_match(label) and confidence > 0.15
        for (_, label, confidence) in decoded
    )
    top_label = decoded[0][1].replace('_', ' ').title()
    top_confidence = decoded[0][2] * 100

    return is_car, top_label, top_confidence

# Load model and results before header (header displays live stats)
try:
    model, selected_features = load_model()
    results = load_results()
    model_loaded = True
except:
    model_loaded = False
    results = {}

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style='border-bottom: 1px solid #2A2D35; padding-bottom: 16px; margin-bottom: 24px;'>
    <div style='display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap;'>
        <div>
            <h1 style='margin:0; font-size:28px; font-weight:600; color:#E8E8E8;'>Car Valuation Engine</h1>
            <p style='margin:4px 0 0; font-size:14px; color:#888888;'>Evolutionary feature selection · XGBoost regression</p>
        </div>
        <div style='text-align:right; font-family:monospace; font-size:11px; color:#666666; line-height:1.6;'>
            NIC 29101<br>AUTOMOTIVE ENGINEERING
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.warning("⚠️ Model not trained yet. Please run train_model.py first.")
else:
    spec1, spec2, spec3, spec4 = st.columns(4)
    spec1.markdown("<p style='font-size:12px;color:#666;margin:0;'>MODEL</p><p style='font-size:14px;font-weight:600;margin:0;'>XGBoost + GA</p>", unsafe_allow_html=True)
    spec2.markdown(f"<p style='font-size:12px;color:#666;margin:0;'>TEST R²</p><p style='font-size:14px;font-weight:600;margin:0;'>{results['test_r2']:.4f}</p>", unsafe_allow_html=True)
    spec3.markdown("<p style='font-size:12px;color:#666;margin:0;'>DATASET</p><p style='font-size:14px;font-weight:600;margin:0;'>301 records</p>", unsafe_allow_html=True)
    spec4.markdown("<p style='font-size:12px;color:#666;margin:0;'>STATUS</p><p style='font-size:14px;font-weight:600;color:#4CAF50;margin:0;'>Live</p>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top:16px; border-color:#2A2D35;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict Price", "📊 Model Performance", "📈 Data Insights"])

# ══════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════
with tab1:
    st.subheader("Enter Car Details")

    img_col, form_col = st.columns([1, 2])

    with img_col:
        uploaded_image = st.file_uploader("📷 Upload Car Image (optional)",
                                           type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Your Car", use_container_width=True)

            with st.spinner("🔍 Verifying image..."):
                is_car, top_label, confidence = check_is_car(uploaded_image)

            if is_car:
                st.success(f"✅ Car detected: **{top_label}** ({confidence:.1f}% confidence)")
            else:
                st.warning(f"⚠️ This doesn't look like a car (detected: **{top_label}**, {confidence:.1f}%).")
        else:
            st.info("No image uploaded yet")

    with form_col:
        col1, col2, col3 = st.columns(3)

        with col1:
            present_price = st.number_input("Present Showroom Price (Lakhs)",
                                             min_value=0.5, max_value=100.0, value=5.0, step=0.5)
            kms_driven = st.number_input("Kilometers Driven",
                                          min_value=100, max_value=500000, value=30000, step=1000)

        with col2:
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
            seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])

        with col3:
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            year = st.slider("Year of Purchase", 2000, 2024, 2018)
            owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

    fuel_map = {"Petrol": 2, "Diesel": 0, "CNG": 1}
    seller_map = {"Dealer": 0, "Individual": 1}
    trans_map = {"Manual": 1, "Automatic": 0}

    car_age = 2024 - year

    input_data = {
        'Present_Price': present_price,
        'Kms_Driven': kms_driven,
        'Fuel_Type': fuel_map[fuel_type],
        'Seller_Type': seller_map[seller_type],
        'Transmission': trans_map[transmission],
        'Owner': owner,
        'Car_Age': car_age
    }

    st.markdown("---")

    can_predict = True
    block_reason = ""

    if uploaded_image is None:
        can_predict = False
        block_reason = "⚠️ Please upload a car image first to enable prediction."
    else:
        is_car_check, _, _ = check_is_car(uploaded_image)
        if not is_car_check:
            can_predict = False
            block_reason = "⚠️ The uploaded image doesn't look like a car. Please upload a valid car photo to proceed."

    if not can_predict:
        st.warning(block_reason)

    if st.button("🔮 Predict Car Price", use_container_width=True, disabled=not can_predict):
        if model_loaded:
            input_df = pd.DataFrame([input_data])
            input_sel = input_df[selected_features]
            prediction = model.predict(input_sel)[0]

            result_col1, result_col2 = st.columns([1, 2])

            with result_col1:
                if uploaded_image is not None:
                    st.image(uploaded_image, caption="Your Car", use_container_width=True)

            with result_col2:
                st.markdown(f"""
                <div style='background:#171A21; border:1px solid #2A2D35;
                            border-radius:8px; padding:24px;'>
                    <p style='color:#888888; font-size:13px; margin:0;'>PREDICTED SELLING PRICE</p>
                    <h1 style='color:#E8E8E8; font-size:2.5rem; margin:8px 0;'>₹ {prediction:.2f} Lakhs</h1>
                    <p style='color:#666666; font-size:12px; margin:0;'>Based on {len(selected_features)} GA-selected features · XGBoost regression</p>
                </div>
                """, unsafe_allow_html=True)

                low = max(0, prediction * 0.9)
                high = prediction * 1.1
                st.info(f"💡 Estimated price range: ₹ {low:.2f} – ₹ {high:.2f} Lakhs")

            st.markdown("---")
            st.subheader("🔍 Similar Cars from Dataset")

            try:
                df_compare = load_data()
                df_compare['Car_Age'] = 2024 - df_compare['Year']

                fuel_match = df_compare[df_compare['Fuel_Type'] == fuel_type]
                if len(fuel_match) >= 3:
                    df_compare = fuel_match

                df_compare['similarity_score'] = (
                    abs(df_compare['Car_Age'] - car_age) * 2 +
                    abs(df_compare['Selling_Price'] - prediction)
                )

                similar_cars = df_compare.nsmallest(5, 'similarity_score')[
                    ['Car_Name', 'Year', 'Selling_Price', 'Driven_kms', 'Fuel_Type', 'Transmission']
                ]
                similar_cars.columns = ['Car Model', 'Year', 'Selling Price (Lakhs)', 'Kms Driven', 'Fuel Type', 'Transmission']

                st.dataframe(similar_cars, use_container_width=True, hide_index=True)
                st.caption("Closest matches from the training dataset based on age and price similarity.")
            except Exception as e:
                st.caption(f"Similar car comparison unavailable: {e}")
        else:
            st.error("Model not loaded. Train the model first.")

# ══════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════
with tab2:
    if model_loaded:
        st.subheader("📊 Model Performance Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Test R² Score", f"{results['test_r2']:.4f}")
        col2.metric("Test MAE", f"{results['test_mae']:.4f} Lakhs")
        col3.metric("GA Best CV R²", f"{results['best_r2_cv']:.4f}")

        st.markdown("---")
        st.subheader("🧬 Genetic Algorithm Convergence")

        if 'convergence_history' in results:
            conv_df = pd.DataFrame(results['convergence_history'])

            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=conv_df['generation'], y=conv_df['best_r2'],
                mode='lines+markers', name='Best R² (this generation)',
                line=dict(color='#4A90D9', width=2)
            ))
            fig_conv.add_trace(go.Scatter(
                x=conv_df['generation'], y=conv_df['overall_best_r2'],
                mode='lines', name='Overall Best R²',
                line=dict(color='#4CAF50', width=3, dash='dash')
            ))
            fig_conv.update_layout(
                title="GA Fitness Improvement Over Generations",
                xaxis_title="Generation",
                yaxis_title="R² Score (Cross-Validated)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_conv, use_container_width=True)
            st.caption("Shows how the Genetic Algorithm progressively found better feature subsets across 20 generations.")

        st.markdown("---")
        st.subheader("✅ Features Selected by Genetic Algorithm")
        for i, feat in enumerate(results['selected_features'], 1):
            st.write(f"**{i}.** {feat}")
        st.markdown("---")
        st.subheader("🖼️ Image Verification Model — Evaluation")
        st.caption("Separate evaluation for the MobileNetV2-based car image classifier (classification task, not regression).")

        try:
            with open("outputs/confusion_matrix_results.json") as f:
                cm_data = json.load(f)

            cm_array = np.array(cm_data["confusion_matrix"])
            labels = cm_data["labels"]

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_array,
                x=[f"Predicted: {l}" for l in labels],
                y=[f"Actual: {l}" for l in labels],
                text=cm_array,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=True
            ))
            fig_cm.update_layout(
                title=f"Confusion Matrix — Image Verification ({cm_data['total_images']} test images)",
                yaxis_autorange='reversed'
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            st.metric("Image Verification Accuracy", f"{cm_data['accuracy']*100:.1f}%")
        except FileNotFoundError:
            st.caption("Run confusion_matrix_test.py to generate this evaluation.")
        st.markdown("---")
        st.subheader("🎯 Feature Importance (XGBoost)")

        if 'feature_importance' in results:
            imp_df = pd.DataFrame(
                list(results['feature_importance'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=True)

            fig_imp = go.Figure(go.Bar(
                x=imp_df['Importance'], y=imp_df['Feature'],
                orientation='h', marker_color='#888888'
            ))
            fig_imp.update_layout(
                title="How Much Each Feature Influences Price Prediction",
                xaxis_title="Importance Score",
                yaxis_title=""
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption("Higher values mean the feature has more influence on the final predicted price.")

# ══════════════════════════════════════════
# TAB 3 — DATA INSIGHTS
# ══════════════════════════════════════════
with tab3:
    try:
        df = load_data()
        st.subheader("📈 Dataset Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("Features", df.shape[1])
        col3.metric("Avg Selling Price", f"₹ {df['Selling_Price'].mean():.2f} L")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(df, x='Selling_Price', nbins=30,
                               title="Distribution of Selling Prices",
                               color_discrete_sequence=['#4A90D9'])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(df, x='Present_Price', y='Selling_Price',
                             color='Fuel_Type', title="Present Price vs Selling Price")
            st.plotly_chart(fig, use_container_width=True)

        fig = px.box(df, x='Fuel_Type', y='Selling_Price',
                     title="Price by Fuel Type", color='Fuel_Type')
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not load data: {e}")