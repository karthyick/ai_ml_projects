import streamlit as st
import requests
import pandas as pd
import random
from pandasai.smart_dataframe import SmartDataframe
from pandasai.llm.google_gemini import GoogleGemini
import warnings
# Ensure correct initialization of Google Gemini LLM
try:
    llm = GoogleGemini(api_key="here")  # Replace with your API key
except ImportError:
    st.error("GoogleGemini module not found. Ensure PandasAI is updated and 'google-generativeai' is installed.")

# Initialize SmartDataframe with config instead of llm directly
config = {"llm": llm}  # Pass LLM in a config dictionary
pandas_ai = SmartDataframe(df=[], config=config)

# Function to fetch historical data from web search
def fetch_historical_data(query):
    api_key = 'here' 
    search_url = f'https://serpapi.com/search.json?q={query}&api_key={api_key}'
    response = requests.get(search_url)
    if response.status_code == 200:
        data = response.json()
        results = data.get("organic_results", [])
        if not results:
            return [{"title": "No results found", "link": "N/A"}]
        return results
    else:
        return [{"title": "Error fetching data", "link": "N/A"}]

# Function to generate alternative history
def generate_alternative_history(df, period):
    prompt = (
    f"Imagine an alternate history where {period} never happened or had a completely different outcome. "
    f"Break down the impacts into the following categories:\n\n"
    f"1️⃣ **Science & Technology**: How would key scientific discoveries, inventions, or technological progress be affected? "
    f"Would new theories replace old ones? Would breakthroughs in medicine, computing, or physics be delayed or accelerated?\n\n"
    f"2️⃣ **Politics & Geopolitics**: How would world power structures shift? Would new alliances form, or conflicts arise? "
    f"How would governments, policies, and international relations change?\n\n"
    f"3️⃣ **Culture & Society**: How would art, literature, and philosophy be influenced? Would social structures and norms evolve differently? "
    f"What cultural trends would emerge or disappear?\n\n"
    f"4️⃣ **Daily Life**: How would the lives of ordinary people be different? Would their jobs, communication, or education change? "
    f"How would daily routines and lifestyles be affected?\n\n"
    f"Provide a detailed and imaginative response that explores this alternative reality in depth."
    )
    try:
        df = df.applymap(lambda x: str(x) if isinstance(x, (list, dict)) else x)  # Convert unhashable lists/dicts to strings
        smart_df = SmartDataframe(df.copy(), config={"llm": llm})  # Ensure a copy is passed
        response = smart_df.chat(prompt)
        if not response or response.strip() == "No code found in the response":
            return "Sorry, I couldn't generate an alternative history for this period. Please try a different query."
        return response
    except Exception as e:
        return f"Error generating alternative history: {str(e)}"

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Time Traveler", layout="wide")
    st.title("⏳ AI-Powered Time Traveler")
    
    query = st.text_input("Search for a Historical Period:", "Ancient History")
    if "events" not in st.session_state:
        st.session_state.events = []
    
    if st.button("🔍 Fetch Events"):
        with st.spinner("Fetching historical events..."):
            st.session_state.events = fetch_historical_data(query)
        
        st.write("### 📜 AI-Fetched Historical Events:")
        for event in st.session_state.events[:5]:  # Display top 5 results
            st.write(f"- {event.get('title', 'Unknown Event')}: {event.get('link', '')}")
    
    if st.button("🌀 Generate Alternative History"):
        if not st.session_state.events or st.session_state.events[0]["title"] in ["No results found", "Error fetching data"]:
            st.warning("Please fetch valid events first before generating alternative history.")
        else:
            with st.spinner("Rewriting history..."):
                df = pd.DataFrame(st.session_state.events)
                df = df.applymap(lambda x: str(x) if isinstance(x, (list, dict)) else x)  # Convert unhashable lists/dicts to strings
                alt_history = generate_alternative_history(df, query)
            st.write("### 🤯 What If Scenario:")
            st.write(alt_history)
    
if __name__ == "__main__":
    main()
