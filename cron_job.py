import pandas as pd
import requests
import json
from datetime import datetime
# (Import your ML libraries here as well)

def update_data():
    url = 'https://data.cityofnewyork.us/resource/i4gi-tjb9.json?$limit=2000'
    response = requests.get(url).json()
    df = pd.DataFrame(response)
    
    # 1. Clean & Calculate (Put your logic here)
    avg_speed = pd.to_numeric(df['speed']).mean()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 2. Create the "Final Result" dictionary
    result = {
        "last_updated": current_time,
        "avg_speed": round(avg_speed, 2),
        "status": "Green" if avg_speed > 25 else "Red"
    }
    
    # 3. Save as a JSON file
    with open('traffic_data.json', 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    update_data()
