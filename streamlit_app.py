import streamlit as st
import json

st.title("NYC Traffic Dashboard (Cloud Updated)")

# Read the pre-calculated data
with open('traffic_data.json', 'r') as f:
    data = json.load(f)

st.metric("Average Speed", f"{data['avg_speed']} mph")
st.write(f"Last updated by Cloud: {data['last_updated']}")

if data['status'] == "Red":
    st.error("Heavy Traffic Detected")
else:
    st.success("Traffic is Moving Well")
