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
# Claude API (Modify for your org's setup)
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "your-claude-api-key")
CLAUDE_ENDPOINT = "https://api.anthropic.com/v1/complete"

CLAUDE_MODELS = {
    "Claude-2": "claude-2",
    "Claude-3": "claude-3"
}


def get_azure_gpt_response(model_key, prompt):
    model_info = AZURE_GPT_MODELS[model_key]
    
    llm = AzureChatOpenAI(
        deployment_name=model_info["deployment_name"],
        model_name=model_info["model_name"],
        temperature=0
    )

    start_time = time.time()
    response = llm.invoke(prompt)
    end_time = time.time()

    latency = round(end_time - start_time, 2)
    result = response.content if response else "No response"

    return model_key, result, latency

# Function to get response from Claude models via API
def get_claude_response(model_key, prompt):
    endpoint = CLAUDE_ENDPOINTS[model_key]
    
    headers = {
        "Authorization": f"Bearer {CLAUDE_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model_key,
        "prompt": prompt,
        "temperature": 0,
    }

    start_time = time.time()
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        end_time = time.time()
        latency = round(end_time - start_time, 2)

        if response.status_code == 200:
            result = response.json().get("content", "No response")
        else:
            result = f"Error: {response.status_code} - {response.text}"
    
    except Exception as e:
        result = f"Error: {str(e)}"
        latency = "N/A"

    return model_key, result, latency

# Streamlit UI setup
st.set_page_config(layout="wide")
st.title("🔍 AI Model Performance Comparison (Azure GPT & Claude)")

# Input prompt at the bottom
st.markdown("---")
prompt = st.text_area("Enter your prompt:", "How to make tea?", height=100)

if st.button("Run Prompt"):
    st.subheader("Generating Responses...")

    results = []

    # Fetch responses from Azure GPT models
    for model in AZURE_GPT_MODELS.keys():
        results.append(get_azure_gpt_response(model, prompt))

    # Fetch responses from Claude models
    for model in CLAUDE_ENDPOINTS.keys():
        results.append(get_claude_response(model, prompt))

    # Sort responses based on latency (fastest first)
    results.sort(key=lambda x: x[2] if isinstance(x[2], float) else float("inf"))

    # Display results in columns
    columns = st.columns(len(results))

    for col, (model_name, response_text, latency) in zip(columns, results):
        col.markdown(f"### {model_name} (⏳ {latency} sec)")
        col.text_area("Response", response_text, height=300, key=model_name)


def get_claude_response(model_key, prompt):
    endpoint = CLAUDE_ENDPOINTS[model_key]
    
    headers = {
        "Authorization": f"Bearer {CLAUDE_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model_key,
        "prompt": prompt,
        "temperature": 0,
    }

    start_time = time.time()
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        end_time = time.time()
        latency = round(end_time - start_time, 2)

        if response.status_code == 200:
            result = response.json().get("content", "No response")
        else:
            result = f"Error: {response.status_code} - {response.text}"
    
    except Exception as e:
        result = f"Error: {str(e)}"
        latency = "N/A"

    return model_key, result, latency

# Streamlit UI
st.title("Claude Models Response Comparison")

# Input prompt at the bottom
st.markdown("---")
prompt = st.text_area("Enter your prompt:", "How to make tea?", height=100)

if st.button("Run Prompt"):
    st.subheader("Generating Responses...")

    results = []
    
    # Fetch responses for each Claude model
    for model in CLAUDE_ENDPOINTS.keys():
        results.append(get_claude_response(model, prompt))
    
    # Sort responses based on latency (fastest first)
    results.sort(key=lambda x: x[2] if isinstance(x[2], float) else float("inf"))

    # Display results in columns
    columns = st.columns(len(results))

    for col, (model_name, response_text, latency) in zip(columns, results):
        col.markdown(f"### {model_name} (⏳ {latency} sec)")
        col.text_area("Response", response_text, height=200, key=model_name)

def get_claude_response(model_key, prompt, output_placeholder):
    llm = ChatAnthropic(
        model=CLAUDE_MODELS[model_key],
        temperature=0,
        timeout=None,
        max_retries=2,
        anthropic_api_key=ANTHROPIC_API_KEY
    )

    start_time = time.time()
    messages = [("system", "You are a helpful assistant"), ("human", prompt)]
    response = llm.invoke(messages)
    end_time = time.time()

    latency = round(end_time - start_time, 2)
    result = response.content if response else "No response"

    # Update UI with model response
    output_placeholder.markdown(f"### {model_key} (⏳ {latency} sec)")
    output_placeholder.text_area("Response", result, height=200, key=model_key)
# def get_claude_response(model_key, prompt, output_placeholder):
#     start_time = time.time()

#     headers = {
#         "x-api-key": CLAUDE_API_KEY,
#         "anthropic-version": "2023-06-01",
#         "content-type": "application/json"
#     }
#     payload = {
#         "model": CLAUDE_MODELS[model_key],
#         "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
#         "max_tokens_to_sample": 500
#     }

#     response = requests.post(CLAUDE_ENDPOINT, json=payload, headers=headers)
#     end_time = time.time()

#     latency = round(end_time - start_time, 2)

#     if response.status_code == 200:
#         result = response.json().get("completion", "No response")
#     else:
#         result = f"Error: {response.text}"

#     # Update UI with model response
#     output_placeholder.markdown(f"### {model_key} (⏳ {latency} sec)")
#     output_placeholder.text_area("Response", result, height=200, key=model_key)

# Streamlit UI Layout
st.set_page_config(layout="wide")  # Expand layout width
st.title("Multi-Model AI Chat (Azure OpenAI & Claude)")

# **Create a horizontal layout with columns**
col1, col2, col3, col4, col5 = st.columns(5)

# **Create placeholders inside columns**
output_placeholders = {
    "GPT-3": col1.empty(),
    "GPT-3.5": col2.empty(),
    "GPT-3.5-Turbo": col3.empty(),
    "Claude-2": col4.empty(),
    "Claude-3": col5.empty(),
}

# **Prompt input at the bottom**
st.markdown("---")  # Horizontal line separator
prompt = st.text_area("Enter your prompt:", "Explain black coffee.", height=100)

# **Run button**
if st.button("Run Prompt"):
    st.subheader("Generating Responses...")

    # **Trigger all models in parallel**
    for model in MODEL_CONFIGS.keys():
        get_azure_response(model, prompt, output_placeholders[model])

    for model in CLAUDE_MODELS.keys():
        get_claude_response(model, prompt, output_placeholders[model])


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
# # **Update UI as soon as result is available**
#     output_placeholder.markdown(f"### {model_key} (⏳ {latency} sec)")
#     output_placeholder.text_area("Response", result.content, height=200, key=model_key)

# # Streamlit UI
# st.set_page_config(layout="wide")  # Expands page width
# st.title("Azure OpenAI Multi-Model Chat (Streaming Mode)")
# st.write("Compare responses & latencies across multiple models (fastest response appears first).")

# # User input prompt
# prompt = st.text_area("Enter your prompt:", "Explain black coffee in simple terms.", height=150)

# if st.button("Run Prompt"):
#     st.subheader("Results")

#     # **Create placeholders for each model**
#     output_placeholders = {model: st.empty() for model in MODEL_CONFIGS.keys()}

#     # **Trigger models in parallel and update UI as they finish**
#     for model in MODEL_CONFIGS.keys():
#         get_response(model, prompt, output_placeholders[model])

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
