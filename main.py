import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(
    page_title="Soil Erosion Prediction",
    page_icon="🌱",
    layout="wide"
)

# -----------------------------------------------------------
# LOAD TRAINED MODEL
# -----------------------------------------------------------
@st.cache_resource
def load_model(pickle_file="soil_erosion_best_model.pkl"):
    """
    Load the trained model saved using joblib.
    """
    model = joblib.load(pickle_file)
    return model

model = load_model()

# -----------------------------------------------------------
# MAIN UI LAYOUT
# -----------------------------------------------------------
st.title("🌱 Soil Erosion Prediction System")
st.markdown("### Predict soil erosion using Machine Learning")

st.write(
    """
    Enter the required soil and environmental parameters in the sidebar to predict **soil erosion values**.
    """
)

st.sidebar.header("🔧 Input Parameters")

# -----------------------------------------------------------
# USER INPUT SECTION
# -----------------------------------------------------------
# Sliders/inputs for all 9 features
MAT = st.sidebar.slider("Mean Annual Temperature (°C)", -10.0, 50.0, 20.0, 0.1)
MAP = st.sidebar.slider("Mean Annual Precipitation (mm)", 0.0, 5000.0, 1000.0, 1.0)
Elevation = st.sidebar.slider("Elevation (m)", 0.0, 5000.0, 250.0, 1.0)
Latitude = st.sidebar.slider("Latitude (°)", -90.0, 90.0, 20.0, 0.01)
Longitude = st.sidebar.slider("Longitude (°)", -180.0, 180.0, 80.0, 0.01)
Slope = st.sidebar.slider("Slope (%)", 0.0, 100.0, 15.0, 0.1)
Soil_sand = st.sidebar.slider("Soil Sand Content (%)", 0.0, 100.0, 40.0, 0.1)
Soil_silt = st.sidebar.slider("Soil Silt Content (%)", 0.0, 100.0, 40.0, 0.1)
Soil_clay = st.sidebar.slider("Soil Clay Content (%)", 0.0, 100.0, 20.0, 0.1)

# Convert single prediction input to DataFrame
input_df = pd.DataFrame([{
    'MAT': MAT,
    'MAP': MAP,
    'Elevation': Elevation,
    'Latitude': Latitude,
    'Longitude': Longitude,
    'Slope': Slope,
    'Soil_sand': Soil_sand,
    'Soil_silt': Soil_silt,
    'Soil_clay': Soil_clay
}])

# -----------------------------------------------------------
# PREDICTION BUTTON
# -----------------------------------------------------------
if st.sidebar.button("Predict Soil Erosion"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"### 🌾 Predicted Soil Erosion Value: **{prediction:.3f}**")
        st.markdown(
            """
            ### 📊 Interpretation  
            - Higher erosion values indicate higher soil degradation risk  
            - Consider improving soil conservation techniques  
            """
        )
    except Exception as e:
        st.error(f"Prediction failed. Please check the pickle file. Error: {e}")

# -----------------------------------------------------------
# BATCH PREDICTION
# -----------------------------------------------------------
# st.markdown("---")
# st.subheader("📁 Upload CSV for Batch Prediction")

# uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

# if uploaded:
#     try:
#         df = pd.read_csv(uploaded)
#         df.columns = df.columns.str.strip()  # remove leading/trailing spaces
#         st.write("### Uploaded Data Preview")
#         st.dataframe(df, use_container_width=True)

#         required_cols = ['MAT','MAP','Elevation','Latitude','Longitude','Slope','Soil_sand','Soil_silt','Soil_clay']
#         missing_cols = [col for col in required_cols if col not in df.columns]
#         if missing_cols:
#             st.error(f"Missing columns in CSV: {missing_cols}")
#         else:
#             # Ensure numeric
#             df_batch = df[required_cols].apply(pd.to_numeric, errors='coerce')
#             if df_batch.isnull().any().any():
#                 st.error("CSV contains non-numeric values in required columns.")
#             else:
#                 batch_pred = model.predict(df_batch)
#                 df["Predicted_Soil_Erosion"] = batch_pred
#                 st.success("✅ Batch Prediction Completed!")
#                 st.dataframe(df, use_container_width=True)

#                 # Download option
#                 csv = df.to_csv(index=False).encode("utf-8")
#                 st.download_button("Download Predictions CSV", csv, "predictions.csv")
#     except Exception as e:
#         st.error(f"Batch prediction failed. Error: {e}")

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown(
    """
    <hr>
    <center>Developed as part of Minor Project — Soil Erosion Prediction ML System 🌍</center>
    """,
    unsafe_allow_html=True
)
