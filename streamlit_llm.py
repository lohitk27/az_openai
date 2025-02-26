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


def get_response(model_key, prompt, output_placeholder):
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



results_container = st.container()

# **User input at the bottom**
with st.container():
    prompt = st.text_area("Enter your prompt:", "Explain black coffee in simple terms.")
    run_button = st.button("Run Prompt")

# **Process the prompt when button is clicked**
if run_button and prompt:
    with results_container:
        st.subheader("Results (Sorted by Fastest Response)")
        
        results = []
        for model in MODEL_CONFIGS.keys():
            response, latency = get_response(model, prompt)
            results.append({"model": model, "response": response, "latency": latency})

        # **Sort results by latency (fastest first)**
        results = sorted(results, key=lambda x: x["latency"])

        # **Arrange responses in 2 rows with 3 models each**
        col1, col2, col3 = st.columns(3)  # First row
        col4, col5, col6 = st.columns(3)  # Second row

        columns = [col1, col2, col3, col4, col5, col6]

        for i, result in enumerate(results):
            with columns[i]:  # Assign each model to a separate column
                st.write(f"### {result['model']} (⏳ {result['latency']} sec)")
                st.text_area("Response", result["response"], height=300)
# below code will work based on top latency
# st.set_page_config(layout="wide")  # Ensure this is at the top

# st.title("Azure OpenAI Multi-Model Chat (Sorted by Latency)")
# st.write("Compare responses & latencies across multiple models (fastest first).")

# # **Create a container for results (this will be displayed above input)**
# results_container = st.container()

# # **User input at the bottom**
# with st.container():
#     prompt = st.text_area("Enter your prompt:", "Explain black coffee in simple terms.")
#     run_button = st.button("Run Prompt")

# # **Process the prompt when button is clicked**
# if run_button and prompt:
#     with results_container:
#         st.subheader("Results")
#         st.write("Fetching responses...")

#         results = []

#         # **Call different models**
#         for model in MODEL_CONFIGS.keys():
#             response, latency = get_response(model, prompt)
#             results.append({"model": model, "response": response, "latency": latency})

#         # **Sort results by latency (fastest first)**
#         results = sorted(results, key=lambda x: x["latency"])

#         # **Display results above input**
#         for result in results:
#             st.write(f"### {result['model']}")
#             st.text_area("Response", result["response"], height=200)
#             st.write(f"⏳ Latency: {result['latency']} seconds")
