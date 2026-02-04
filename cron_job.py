import pandas as pd
import requests
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime

def run_accurate_update():
    url = 'https://data.cityofnewyork.us/resource/i4gi-tjb9.json?$limit=200'
    try:
        response = requests.get(url).json()
        df = pd.DataFrame(response)
        
        # --- YOUR WORKING CLEANING LOGIC ---
        df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
        df['data_as_of'] = pd.to_datetime(df['data_as_of'])
        df['hour'] = df['data_as_of'].dt.hour
        df['day_of_week'] = df['data_as_of'].dt.dayofweek
        df['link_id'] = pd.to_numeric(df['link_id'], errors='coerce')
        df = df.dropna(subset=['speed', 'hour', 'day_of_week', 'link_id'])

        # Feature Creation
        df_grouped = df.groupby(['link_id', 'hour', 'day_of_week']).agg({'speed': 'mean'}).reset_index()
        df_grouped['speed_lag_1h'] = df_grouped.groupby('link_id')['speed'].shift(1)
        df_grouped = df_grouped.dropna()

        if len(df_grouped) > 50:
            X = df_grouped[['hour', 'day_of_week', 'speed_lag_1h']]
            y = df_grouped['speed']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # --- YOUR WORKING MODEL ---
            model = GradientBoostingRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            accuracy = (1 - mae/y_test.mean()) * 100

            # Current Speed Analysis
            current_hour = datetime.now().hour
            current_speeds = df[df['hour'] == current_hour]['speed']
            avg_speed = current_speeds.mean() if len(current_speeds) > 0 else df['speed'].mean()
            
            # --- THE OUTPUT FOR YOUR IPHONE ---
            result = {
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M'),
                "avg_speed": round(float(avg_speed), 1),
                "accuracy": round(float(accuracy), 1),
                "status": "🔴 HEAVY" if avg_speed < 15 else "🟡 MODERATE" if avg_speed < 25 else "🟢 LIGHT"
            }

            with open('traffic_data.json', 'w') as f:
                json.dump(result, f)
            print("✓ Success: traffic_data.json created.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_accurate_update()
