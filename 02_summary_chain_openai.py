
"""
===============================================================================
Future with Ayesha — AI Harness
Section 1 | Script 02: Building a Summary Chain with LangChain and Groq
===============================================================================

OVERVIEW
--------
This script demonstrates how to build a reusable LLM-powered summarization
chain using LangChain Expression Language (LCEL) and Groq.

Given a biography, the chain generates:
    1. A concise headline.
    2. A three-bullet summary suitable for a 12-year-old.
    3. One interview question based on the supplied information.

LEARNING OBJECTIVES
-------------------
1. Create reusable prompts with PromptTemplate.
2. Connect LangChain to a Groq-hosted chat model using ChatGroq.
3. Combine components using the LCEL pipe operator (|).
4. Execute a chain with invoke().
5. Configure optional LangSmith tracing through environment variables.

WORKFLOW
--------
    Biography
        |
        v
    PromptTemplate
        |
        v
    ChatGroq
        |
        v
    AIMessage
        |
        v
    Generated summary

PREREQUISITES
-------------
Install the dependencies:

    python3 -m pip install langchain-core langchain-groq python-dotenv

Create a .env file in this directory:

    GROQ_API_KEY=your_groq_api_key
    GROQ_MODEL=openai/gpt-oss-120b

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=your_langsmith_api_key
    LANGSMITH_PROJECT=section-01-langchain-foundations

The model setting is optional. The default is configured below, but model
availability and account access may change. Check the Groq Console if needed.

For compatibility, OPENAI_API_KEY is accepted as a fallback for an existing
Groq key stored under that variable name. The key is still sent to Groq.

SECURITY
--------
Never hardcode API keys or commit the .env file to version control.
Add .env to .gitignore. Do not print or share secret credentials.

INPUT DATA
----------
PERSON_INFORMATION is imported from sample_data.py. Keeping the biography
separate from the chain makes it easier to reuse the workflow with other text.

HOW TO RUN
----------
    python3 02_summary_chain_openai.py

The original filename is retained for compatibility with the exercise, although
the model integration in this version uses Groq rather than OpenAI.

EXPECTED OUTPUT
---------------
The script prints the greeting, selected model name, and generated summary.
The exact wording depends on the model response.

===============================================================================
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from sample_data import PERSON_INFORMATION


# Load .env from the script's directory. Existing shell environment variables
# take precedence over values in the file.
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_DIR / ".env")


# Read the model name from the environment, using a default if not configured.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# Define the reusable prompt. The {information} placeholder is filled when
# the chain is invoked with the biography.
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
    Build a reusable biography summarization chain.

    Args:
        model_name: A model identifier supported by the Groq API.

    Returns:
        A LangChain runnable sequence consisting of a prompt and chat model.

    The LCEL pipe operator connects PromptTemplate to ChatGroq, passing the
    formatted prompt to the model when the chain is executed.
    """

    # Define the prompt's required input variable.
    prompt = PromptTemplate(
        input_variables=["information"],
        template=PROFILE_TEMPLATE,
    )

    # Prefer GROQ_API_KEY. The fallback supports an existing configuration
    # where the Groq key is stored under OPENAI_API_KEY.
    groq_api_key = (
        os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    if not groq_api_key:
        raise EnvironmentError(
            "Groq API key is missing. Add GROQ_API_KEY to your .env file."
        )

    # Initialize the Groq-hosted chat model.
    llm = ChatGroq(
        model=model_name,
        temperature=0,
        api_key=groq_api_key,
    )

    # Compose the prompt and model into an executable LCEL chain.
    return prompt | llm


def main() -> None:
    """Execute the summarization chain and print the generated response."""

    print("Hello from Future with Ayesha — AI Harness!")
    print(f"Model: {GROQ_MODEL}\n")

    # Construct the chain.
    chain = build_profile_chain(GROQ_MODEL)

    # The dictionary key must match the {information} prompt placeholder.
    response = chain.invoke(
        {"information": PERSON_INFORMATION}
    )

    # AIMessage.text returns the generated response as a string.
    print(response.content)


if __name__ == "__main__":
    # Run the example only when this file is executed directly.
    main()
