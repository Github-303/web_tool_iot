import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

from langchain_openai import ChatOpenAI
from src.logger.base import BaseLogger
from src.models.llms import load_llm
from src.utils import execute_plt_code
load_dotenv()
logger = BaseLogger()
MODEL_NAME = "gpt-3.5-turbo"

def process_query(da_agent, query):
    response = da_agent(query)
    # Check if the response contains intermediate steps
    if "intermediate_steps" in response and response["intermediate_steps"]:
        last_step = response["intermediate_steps"][-1]
        if isinstance(last_step, tuple) and len(last_step) > 0:
            action = last_step[0].tool_input if hasattr(last_step[0], 'tool_input') else None
            if action and isinstance(action, dict) and "query" in action:
                action = action["query"]
            else:
                action = str(action)
    else:
        action = ""

    if "plt" in action:
        st.write(response["output"])
        fig = execute_plt_code(action, st.session_state.df)
        if fig:
            st.pyplot(fig)
    else:
        st.write(response["output"])
        st.session_state.history.append((query, response["output"]))

def main():
    st.set_page_config(
        page_title="IoT Detection",
        page_icon="📊",
        layout="centered",
    )
    
    # Configure server settings for large file uploads
    st.config.set_option('server.maxUploadSize', 300)
    st.config.set_option('server.enableCORS', False)
    st.config.set_option('server.enableXsrfProtection', False)
    
    st.header("📊 IoT Detection Tool")
    st.write("### Detect IoT Malware")
    st.write("This is a tool to detect IoT Malware")

    llm = load_llm(model_name=MODEL_NAME)
    logger.info(f"Loaded LLM: {MODEL_NAME}")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if "history" not in st.session_state:
        st.session_state.history = []

    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write("Your uploaded file:", st.session_state.df.head())

        da_agent = create_pandas_dataframe_agent(
            llm = llm, 
            df = st.session_state.df, 
            agent_type="openai-functions",
            verbose=True,
            return_intermediate_steps=True
        )
        logger.info("###Created Pandas DataFrame Agent")

        query = st.text_input("Enter a question about your data:")
        if st.button("Submit"):
            with st.spinner("Processing..."):
                process_query(da_agent, query)
if __name__ == "__main__":
    main()