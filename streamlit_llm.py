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
