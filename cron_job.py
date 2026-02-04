import pandas as pd
import requests
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import os

def run_accurate_update():
    print(f"🚀 Starting update at: {datetime.now()}")
    url = 'https://data.cityofnewyork.us/resource/i4gi-tjb9.json?$limit=400'
    
    try:
        print("📡 Step 1: Fetching data from NYC API...")
        response = requests.get(url).json()
        df = pd.DataFrame(response)
        print(f"✅ Loaded {len(df)} raw records.")
        
        print("🧹 Step 2: Cleaning and formatting data...")
        df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
        df['data_as_of'] = pd.to_datetime(df['data_as_of'])
        df['hour'] = df['data_as_of'].dt.hour
        df['day_of_week'] = df['data_as_of'].dt.dayofweek
        df['link_id'] = pd.to_numeric(df['link_id'], errors='coerce')
        df = df.dropna(subset=['speed', 'hour', 'day_of_week', 'link_id'])
        print(f"✅ Data cleaned. {len(df)} records remaining.")

        print("🧠 Step 3: Training Gradient Boosting Model...")
        df_grouped = df.groupby(['link_id', 'hour', 'day_of_week']).agg({'speed': 'mean'}).reset_index()
        df_grouped['speed_lag_1h'] = df_grouped.groupby('link_id')['speed'].shift(1)
        df_grouped = df_grouped.dropna()

        if len(df_grouped) > 50:
            X = df_grouped[['hour', 'day_of_week', 'speed_lag_1h']]
            y = df_grouped['speed']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = GradientBoostingRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
            model.fit(X_train, y_train)
            
            mae = mean_absolute_error(y_test, model.predict(X_test))
            accuracy = (1 - mae/y_test.mean()) * 100
            print(f"✅ Model trained. Accuracy: {accuracy:.1f}%")

            print("📊 Step 4: Analyzing current conditions...")
            avg_speed = df['speed'].mean()
            
            result = {
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M'),
                "avg_speed": round(float(avg_speed), 1),
                "accuracy": round(float(accuracy), 1),
                "status": "🔴 HEAVY" if avg_speed < 15 else "🟡 MODERATE" if avg_speed < 25 else "🟢 LIGHT"
            }

            print("💾 Step 5: Saving traffic_data.json...")
            with open('traffic_data.json', 'w') as f:
                json.dump(result, f)
            print("✨ Process complete! File saved.")
        else:
            print("⚠️ Error: Not enough data grouped to train.")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")
        exit(1) # This tells GitHub Actions that the run failed

if __name__ == "__main__":
    run_accurate_update()
