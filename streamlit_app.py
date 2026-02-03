import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime

st.set_page_config(page_title="NYC Traffic Predictor", page_icon="🚦")

st.title("🚦 NYC Traffic Speed Prediction")
st.markdown("Fetching live data and training a Gradient Boosting model in real-time.")

# --- STEP 1: Fetch Data ---
with st.status("Fetching live traffic data from NYC API...", expanded=True) as status:
    url = 'https://data.cityofnewyork.us/resource/i4gi-tjb9.json?$limit=2000'
    try:
        data = requests.get(url).json()
        df = pd.DataFrame(data)
        st.write(f"✅ Loaded {len(df)} traffic records")
        
        # --- STEP 2: Clean Data ---
        df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
        df['data_as_of'] = pd.to_datetime(df['data_as_of'])
        df['hour'] = df['data_as_of'].dt.hour
        df['day_of_week'] = df['data_as_of'].dt.dayofweek
        df['link_id'] = pd.to_numeric(df['link_id'], errors='coerce')
        df = df.dropna(subset=['speed', 'hour', 'day_of_week', 'link_id'])
        
        # --- STEP 3: Features ---
        df_grouped = df.groupby(['link_id', 'hour', 'day_of_week']).agg({'speed': 'mean'}).reset_index()
        df_grouped['speed_lag_1h'] = df_grouped.groupby('link_id')['speed'].shift(1)
        df_grouped = df_grouped.dropna()
        status.update(label="Data processed!", state="complete", expanded=False)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

if len(df_grouped) > 50:
    # --- STEP 4 & 5: Model Training ---
    X = df_grouped[['hour', 'day_of_week', 'speed_lag_1h']]
    y = df_grouped['speed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # --- STEP 6: Metrics ---
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    accuracy = (1 - mae/y_test.mean()) * 100

    # --- STEP 7: Current Status ---
    current_hour = datetime.now().hour
    current_speeds = df[df['hour'] == current_hour]['speed']
    avg_speed = current_speeds.mean() if len(current_speeds) > 0 else df['speed'].mean()

    # --- UI DISPLAY ---
    st.header("🎯 Live Traffic Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Speed", f"{avg_speed:.1f} mph")
    col2.metric("Model Accuracy", f"{accuracy:.1f}%")
    col3.metric("Error Margin", f"±{mae:.2f} mph")

    if avg_speed < 15:
        st.error("🔴 TRAFFIC STATUS: HEAVY TRAFFIC - Expect significant delays!")
    elif avg_speed < 25:
        st.warning("🟡 TRAFFIC STATUS: MODERATE TRAFFIC - Some congestion expected")
    else:
        st.success("🟢 TRAFFIC STATUS: LIGHT TRAFFIC - Good driving conditions")
    
    st.subheader("Data Preview")
    st.dataframe(df.head(10))

else:
    st.error("Not enough data available. Please try again in a few minutes.")
