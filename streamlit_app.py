import streamlit as st
import json
import os
import datetime

st.set_page_config(page_title="Debug Dashboard", page_icon="🔍")

st.title("🚗 App Process Tracker")
st.write("---")

# --- DEBUG SECTION ---
st.subheader("🛠️ System Diagnostics")
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.write(f"**Current App Time:** {current_time}")
st.write(f"**Working Directory:** `{os.getcwd()}`")

# List all files to see if the Robot's file is actually there
all_files = os.listdir('.')
st.write("**Files found in folder:**")
st.code(all_files)

# --- DATA LOADING PROCESS ---
st.subheader("📂 Data Loading Process")
DATA_FILE = 'traffic_data.json'

if DATA_FILE in all_files:
    st.success(f"✅ Found {DATA_FILE}!")
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        
        st.write("📊 **Data parsed successfully:**")
        st.json(data) # This prints the raw data for us to see
        
        # Display as metrics
        st.write("---")
        col1, col2 = st.columns(2)
        col1.metric("Predicted Speed", f"{data.get('avg_speed')} mph")
        col2.metric("ML Accuracy", f"{data.get('accuracy')}%")
        
    except Exception as e:
        st.error(f"❌ Failed to read JSON: {e}")
else:
    st.error(f"❌ {DATA_FILE} is missing from the folder.")
    st.info("💡 Tip: If you see the file on GitHub but not in the list above, you must REBOOT the app in Streamlit Cloud.")

if st.button('🔄 Refresh Process'):
    st.rerun()
