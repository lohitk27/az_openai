import streamlit as st
import os
import time
from langchain.chat_models import AzureChatOpenAI

# Load environment variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-azure-openai-endpoint.openai.azure.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-azure-api-key")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-02-01")

# Azure OpenAI Model Configurations
MODEL_CONFIGS = {
    "GPT-3": {"deployment_name": "css-gpt3", "model_name": "gpt3"},
    "GPT-3.5": {"deployment_name": "css-gpt3.5", "model_name": "gpt3.5"},
    "GPT-3.5-Turbo": {"deployment_name": "css-gpt3.5-turbo", "model_name": "gpt3.5-turbo"},
}

# Function to get response from a model
def get_response(model_key, prompt):
    config = MODEL_CONFIGS[model_key]
    
    # Initialize AzureChatOpenAI instance
    llm = AzureChatOpenAI(
        deployment_name=config["deployment_name"],
        model_name=config["model_name"],
        temperature=0
    )

    start_time = time.time()
    result = llm.invoke(prompt)
    end_time = time.time()
    
    latency = round(end_time - start_time, 2)  # Latency in seconds
    return result.content, latency

# Streamlit UI
st.title("Azure OpenAI Multi-Model Chat")
st.write("Compare responses & latencies across multiple models.")

# User input prompt
prompt = st.text_area("Enter your prompt:", "Explain black coffee in simple terms.")

if st.button("Run Prompt"):
    st.subheader("Results")
    
    with st.spinner("Fetching responses..."):
        results = {}
        
        for model in MODEL_CONFIGS.keys():
            response, latency = get_response(model, prompt)
            results[model] = {"response": response, "latency": latency}

    # Display results in separate columns
    col1, col2, col3 = st.columns(3)

    for idx, model in enumerate(results.keys()):
        with [col1, col2, col3][idx]:
            st.write(f"### {model}")
            st.text_area("Response", results[model]["response"], height=200)
            st.write(f"⏳ Latency: {results[model]['latency']} seconds")
# **Update UI as soon as result is available**
    output_placeholder.markdown(f"### {model_key} (⏳ {latency} sec)")
    output_placeholder.text_area("Response", result.content, height=200, key=model_key)

# Streamlit UI
st.set_page_config(layout="wide")  # Expands page width
st.title("Azure OpenAI Multi-Model Chat (Streaming Mode)")
st.write("Compare responses & latencies across multiple models (fastest response appears first).")

# User input prompt
prompt = st.text_area("Enter your prompt:", "Explain black coffee in simple terms.", height=150)

if st.button("Run Prompt"):
    st.subheader("Results")

    # **Create placeholders for each model**
    output_placeholders = {model: st.empty() for model in MODEL_CONFIGS.keys()}

    # **Trigger models in parallel and update UI as they finish**
    for model in MODEL_CONFIGS.keys():
        get_response(model, prompt, output_placeholders[model])

import streamlit as st
import openai
import requests
import time

# Azure OpenAI Configuration (Same API key for all models)
AZURE_OPENAI_ENDPOINT = "https://your-azure-openai-endpoint.openai.azure.com"
API_KEY = "your-azure-api-key"
DEPLOYMENT_GPT3 = "gpt-3-deployment-name"
DEPLOYMENT_GPT3_5 = "gpt-3.5-turbo-deployment-name"
DEPLOYMENT_CLAUDE2 = "claude-2-deployment-name"  # If using Azure's Anthropic model

# Function to query Azure OpenAI GPT models
def query_azure_openai(deployment_name, prompt):
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{deployment_name}/chat/completions?api-version=2024-02-01"
    
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500
    }
    
    start_time = time.time()
    response = requests.post(url, headers=headers, json=data)
    end_time = time.time()
    latency = round(end_time - start_time, 2)  # Response time in seconds
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"], latency
    else:
        return f"Error: {response.json()}", None

# Function to query Claude-2 via Azure (if using Anthropic)
def query_claude_2(prompt):
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }
    url = f"{AZURE_OPENAI_ENDPOINT}/anthropic/deployments/{DEPLOYMENT_CLAUDE2}/chat/completions?api-version=2024-02-01"
    
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500
    }
    
    start_time = time.time()
    response = requests.post(url, headers=headers, json=data)
    end_time = time.time()
    latency = round(end_time - start_time, 2)  # Response time in seconds
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"], latency
    else:
        return f"Error: {response.json()}", None

# Streamlit UI
st.title("Azure OpenAI Multi-Model Chat")
st.write("Enter a prompt to compare responses & latencies across GPT-3, GPT-3.5 Turbo, and Claude-2.")

# User input
prompt = st.text_area("Enter your prompt:", "Explain quantum computing in simple terms.")

if st.button("Run Prompt"):
    st.subheader("Results")
    
    with st.spinner("Fetching responses..."):
        # Query all models
        response_gpt3, latency_gpt3 = query_azure_openai(DEPLOYMENT_GPT3, prompt)
        response_gpt3_5, latency_gpt3_5 = query_azure_openai(DEPLOYMENT_GPT3_5, prompt)
        response_claude2, latency_claude2 = query_claude_2(prompt)

    # Display results in separate chat windows
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### GPT-3")
        st.text_area("Response", response_gpt3, height=200)
        st.write(f"⏳ Latency: {latency_gpt3} seconds")

    with col2:
        st.write("### GPT-3.5 Turbo")
        st.text_area("Response", response_gpt3_5, height=200)
        st.write(f"⏳ Latency: {latency_gpt3_5} seconds")

    with col3:
        st.write("### Claude-2")
        st.text_area("Response", response_claude2, height=200)
        st.write(f"⏳ Latency: {latency_claude2} seconds")
