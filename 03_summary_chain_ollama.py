
"""
===============================================================================
Future with Ayesha — AI Harness
Section 1 | Script 03: Local LLM Summarization with LangChain and Ollama
===============================================================================

OVERVIEW
--------
This script demonstrates how to run a LangChain summarization workflow using
a locally hosted language model through Ollama.

It uses the same prompt structure and LCEL pipeline as Script 02, but replaces
the Groq-hosted model with a local model. This makes it possible to compare
cloud-hosted and locally executed models using a consistent workflow.

Given a biography, the model is asked to generate:
    1. A concise headline.
    2. A three-bullet summary suitable for a 12-year-old.
    3. One interview question based on the supplied information.

LEARNING OBJECTIVES
-------------------
1. Connect LangChain to a local model using ChatOllama.
2. Reuse PromptTemplate and the LCEL pipe operator.
3. Execute a chain with invoke().
4. Configure the local model through environment variables.
5. Understand the trade-offs between local and hosted language models.

WORKFLOW
--------
    Biography
        |
        v
    PromptTemplate
        |
        v
    ChatOllama
        |
        v
    AIMessage
        |
        v
    Generated summary

PREREQUISITES
-------------
1. Install Ollama from https://ollama.com/download.
2. Start the Ollama application or service.
3. Download a supported model:

       ollama pull gemma3:270m

4. Install the Python dependencies:

       python3 -m pip install langchain-core langchain-ollama python-dotenv

Create an optional .env file:

    OLLAMA_MODEL=gemma3:270m

Optional LangSmith tracing:

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=your_langsmith_api_key
    LANGSMITH_PROJECT=section-01-langchain-foundations

The local model does not require a Groq or OpenAI API key. LangSmith tracing
does require its own valid API key when enabled.

HOW TO RUN
----------
    python3 03_summary_chain_ollama.py

The Ollama service must be running and the selected model must be downloaded.

EXPECTED OUTPUT
---------------
The script prints a greeting, the selected local model, and the generated
headline, summary, and interview question.

Output quality and formatting depend on the selected model. Small models may
not consistently follow every instruction.

TROUBLESHOOTING
---------------
Connection errors:
    Start Ollama and verify that its local service is reachable.

Model not found:
    Run `ollama pull <model-name>` using the model configured in OLLAMA_MODEL.

Model quality:
    Try a larger model if your Mac has sufficient memory and storage.

SECURITY AND PRIVACY
--------------------
Local inference sends prompts to the Ollama service on your machine rather
than to a hosted model API. Optional LangSmith tracing can transmit execution
data to LangSmith, so disable tracing if prompts must remain local.

===============================================================================
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from sample_data import PERSON_INFORMATION


# Load environment variables from the .env file beside this script.
# Existing shell environment variables take precedence.
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_DIR / ".env")


# Select the local model. The model must already be available in Ollama.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:270m")


# Use the same prompt structure as Script 02 to make model comparisons fair.
PROFILE_TEMPLATE = """
Here is a short biography:

{information}

Using only the information provided above, produce the following:

1. Headline:
   Write a concise, one-line headline describing the person.

2. Summary for a 12-year-old:
   Provide exactly three simple, informative bullet points.

3. Interview question:
   Suggest one thoughtful question that could be asked in an interview.

Do not introduce facts that are not supported by the supplied biography.
Use clear language and organize the response under the three requested sections.
"""


def build_profile_chain(model_name: str):
    """
    Build a reusable biography summarization chain using a local Ollama model.

    Args:
        model_name: The name of a model available in the local Ollama service.

    Returns:
        A LangChain runnable sequence combining PromptTemplate and ChatOllama.

    The LCEL pipe operator passes the formatted prompt to the local chat model.
    """

    # Define the input variable expected by the prompt template.
    prompt = PromptTemplate(
        input_variables=["information"],
        template=PROFILE_TEMPLATE,
    )

    # Connect to the Ollama service running locally.
    # No Groq or OpenAI API key is required for local inference.
    llm = ChatOllama(
        model=model_name,
        temperature=0,
    )

    # Combine the prompt and model into one executable chain.
    return prompt | llm


def main() -> None:
    """Run the summarization chain and display the generated response."""

    print("Hello from Future with Ayesha — AI Harness!")
    print(f"Model: {OLLAMA_MODEL} (local, via Ollama)\n")

    # Build the chain with the configured local model.
    chain = build_profile_chain(OLLAMA_MODEL)

    # Supply the biography using the key expected by PROFILE_TEMPLATE.
    response = chain.invoke(
        {"information": PERSON_INFORMATION}
    )

    # AIMessage.content contains the generated response text.
    print(response.content)


if __name__ == "__main__":
    # Execute the example only when the script is run directly.
    main()