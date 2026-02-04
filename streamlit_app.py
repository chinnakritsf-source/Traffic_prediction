import streamlit as st
import json
import os

st.set_page_config(page_title="NYC Traffic Live", page_icon="🚗")

st.title("🚗 NYC Live Traffic Prediction")

# This looks for the file you just created!
DATA_FILE = 'traffic_data.json'

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    
    # Show the stats in 3 clean boxes
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Speed", f"{data['avg_speed']} mph")
    col2.metric("Status", data['status'])
    col3.metric("Prediction Accuracy", f"{data['accuracy']}%")

    st.write(f"📅 **Last Updated:** {data['last_updated']}")
    
    # A button to force the app to check for new data
    if st.button('Update Dashboard'):
        st.rerun()
else:
    st.warning("🔄 Connecting to GitHub Data...")
    st.info("The data file was just created! It may take a moment for the website to sync. Please wait 1 minute and refresh.")
