
"""
===============================================================================
Future with Ayesha — AI Harness
Section 1 | Script 04: Inspecting a LangChain AIMessage
===============================================================================

OVERVIEW
--------
This script explores the AIMessage returned by a LangChain model invocation.

In earlier scripts, we printed only the generated text. Here, we inspect the
response object to understand the additional information it contains, including
tool calls, token usage, model metadata, and response identifiers.

These fields become especially useful when building agents and tool-enabled
workflows in later sections.

LEARNING OBJECTIVES
-------------------
1. Understand the AIMessage object returned by a chat model.
2. Inspect response content and generated text.
3. Explore tool_calls and invalid_tool_calls.
4. Read provider-independent token usage metadata.
5. Inspect provider-specific response metadata.
6. Compare cloud-hosted and locally hosted models.

MODEL OPTIONS
-------------
Default:
    Groq using GROQ_MODEL and GROQ_API_KEY.

Optional local model:
    Ollama using OLLAMA_MODEL.

CONFIGURATION
-------------
Add the relevant settings to .env:

    GROQ_API_KEY=your_groq_api_key
    GROQ_MODEL=openai/gpt-oss-120b

    OLLAMA_MODEL=gemma3:270m

Optional LangSmith tracing:

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=your_langsmith_api_key
    LANGSMITH_PROJECT=section-01-langchain-foundations

Keep .env out of version control.

DEPENDENCIES
------------
    python3 -m pip install langchain-core langchain-groq python-dotenv
    python3 -m pip install langchain-ollama

HOW TO RUN
----------
Use Groq:
    python3 04_inspect_the_response_object.py

Use Ollama:
    python3 04_inspect_the_response_object.py --local

For local execution, Ollama must be installed, running, and have the selected
model downloaded.

EXPECTED OUTPUT
---------------
The script displays the response class, message type, a short text preview,
tool calls, token usage, response metadata, model name, and response ID.

Some fields may be empty or unavailable depending on the provider and model.

IMPORTANT
---------
The script inspects a response; it does not automatically calculate API costs.
Token counts and metadata depend on what the provider returns.

===============================================================================
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from sample_data import PERSON_INFORMATION


# Load environment variables from the .env file beside this script.
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_DIR / ".env")


# Configure the default cloud model and optional local model.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:270m")


# The same prompt structure is used for both providers so their responses
# can be inspected using a consistent workflow.
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


def make_llm(use_local: bool):
    """
    Create a chat model using either Groq or Ollama.

    Args:
        use_local: True selects Ollama; False selects Groq.

    Returns:
        A tuple containing the configured chat model and its model name.
    """

    if use_local:
        # Import only when local inference is requested.
        from langchain_ollama import ChatOllama

        return (
            ChatOllama(
                model=OLLAMA_MODEL,
                temperature=0,
            ),
            OLLAMA_MODEL,
        )

    # Prefer GROQ_API_KEY, but support the existing configuration in which
    # a Groq key was saved under OPENAI_API_KEY.
    groq_api_key = (
        os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    if not groq_api_key:
        raise EnvironmentError(
            "Groq API key is missing. Add GROQ_API_KEY to your .env file."
        )

    return (
        ChatGroq(
            model=GROQ_MODEL,
            temperature=0,
            api_key=groq_api_key,
        ),
        GROQ_MODEL,
    )


def inspect_ai_message(response) -> None:
    """
    Print the main fields of a LangChain AIMessage.

    Args:
        response: The AIMessage returned by chain.invoke().
    """

    # response.content contains the generated message content.
    # Convert it to a string safely for the text preview.
    content = response.content

    if isinstance(content, str):
        response_text = content
    else:
        response_text = str(content)

    preview = response_text[:80].replace("\n", "\\n")

    print(f"Response class           : {type(response).__name__}")
    print(f"Message type             : {response.type}")
    print(f"Text preview             : {preview}...")
    print(f"Tool calls               : {response.tool_calls}")
    print(f"Invalid tool calls       : {response.invalid_tool_calls}")
    print(f"Usage metadata            : {response.usage_metadata}")

    # Metadata fields vary between providers.
    metadata = response.response_metadata or {}
    print(f"Response metadata keys    : {list(metadata.keys())}")

    model_name = (
        metadata.get("model_name")
        or metadata.get("model")
        or metadata.get("model_id")
    )
    print(f"Model reported by provider: {model_name}")
    print(f"Response ID               : {response.id}")

    # Content may be a string or a list of structured blocks.
    print(f"Content type              : {type(content).__name__}")

    if isinstance(content, list):
        print("Content contains structured blocks.")

    # Token usage may not be available for every provider.
    if response.usage_metadata is None:
        print("Token usage is not available for this response.")


def main() -> None:
    """Run the chain and inspect the returned AIMessage."""

    use_local = "--local" in sys.argv

    llm, model_name = make_llm(use_local)

    provider = "Ollama (local)" if use_local else "Groq (cloud)"
    print("Hello from Future with Ayesha — AI Harness!")
    print(f"Provider: {provider}")
    print(f"Model: {model_name}\n")
    print("Generating response...\n")

    # PromptTemplate.from_template automatically detects the {information}
    # placeholder. The LCEL pipe operator connects the prompt to the model.
    chain = PromptTemplate.from_template(PROFILE_TEMPLATE) | llm

    # Invoke the chain. The returned AIMessage contains the answer and metadata.
    response = chain.invoke(
        {"information": PERSON_INFORMATION}
    )

    print("=" * 75)
    print("GENERATED RESPONSE")
    print("=" * 75)
    print(response.content)
    print("\n" + "=" * 75)
    print("AIMessage INSPECTION")
    print("=" * 75)

    inspect_ai_message(response)


if __name__ == "__main__":
    main()