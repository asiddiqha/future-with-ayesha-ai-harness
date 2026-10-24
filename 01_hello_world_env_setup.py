
"""
===============================================================================
Future with Ayesha — AI Harness
Section 1 | Script 01: Environment Setup and API Key Validation
===============================================================================

OVERVIEW
--------
This script verifies that Python can load environment variables from a .env
file and checks whether the API credentials required by the project are set.

It is the initial setup step before running the LangChain examples in this
section.

LEARNING OBJECTIVES
-------------------
1. Load environment variables using python-dotenv.
2. Keep API credentials outside the source code.
3. Validate the presence of Groq and LangSmith configuration.
4. Understand how environment variables control optional LangSmith tracing.

CONFIGURATION
-------------
Create a .env file in the project directory.

Recommended configuration:

    GROQ_API_KEY=your_groq_api_key
    GROQ_MODEL=llama-3.3-70b-versatile

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=your_langsmith_api_key
    LANGSMITH_PROJECT=section-01-langchain-foundations

For compatibility with an existing setup, OPENAI_API_KEY is accepted as a
fallback for the Groq key. This does not make it an OpenAI credential; the
value is still used to authenticate with Groq.

SECURITY
--------
- Never hardcode API keys in Python source files.
- Never print API keys or their prefixes.
- Add .env to .gitignore to prevent accidental commits.
- Do not share .env files publicly.

DEPENDENCIES
------------
Install the required package:

    python3 -m pip install python-dotenv

HOW TO RUN
----------
    python3 01_hello_world_env_setup.py

EXPECTED BEHAVIOUR
------------------
The script prints a greeting, reports whether a Groq API key is configured,
and indicates whether LangSmith tracing is enabled.

This script checks configuration only. It does not send an API request or
verify that a key is valid, active, or has sufficient usage quota.

===============================================================================
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# Load environment variables from the project directory's .env file.
# Existing shell environment variables take precedence over values in .env.
PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE)


from typing import Optional

def get_groq_api_key() -> Optional[str]:
    """
    Return the configured Groq API key.

    GROQ_API_KEY is the preferred variable name. OPENAI_API_KEY is supported
    as a temporary fallback for existing setups that store a Groq key under
    that name.

    Returns:
        str | None: The configured key, or None if neither variable is set.
    """
    return (
        os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def describe_groq_configuration() -> str:
    """
    Report whether a Groq API key is configured without revealing its value.

    Returns:
        str: A human-readable configuration status.
    """
    if os.getenv("GROQ_API_KEY"):
        return "Groq API key configuration: found GROQ_API_KEY."

    if os.getenv("OPENAI_API_KEY"):
        return (
            "Groq API key configuration: found OPENAI_API_KEY as a fallback. "
            "Consider renaming it to GROQ_API_KEY."
        )

    return (
        "Groq API key configuration: missing. "
        "Add GROQ_API_KEY to your .env file."
    )


def describe_langsmith_configuration() -> str:
    """
    Report the LangSmith tracing configuration.

    Tracing requires LANGSMITH_TRACING to be enabled and a LangSmith API key
    to be configured. This checks configuration presence, not key validity.
    """
    tracing_enabled = (
        os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true"
    )
    langsmith_key_present = bool(os.getenv("LANGSMITH_API_KEY"))

    if tracing_enabled and langsmith_key_present:
        return "LangSmith tracing: enabled and API key configured."

    if tracing_enabled and not langsmith_key_present:
        return (
            "LangSmith tracing: requested, but LANGSMITH_API_KEY is missing."
        )

    return "LangSmith tracing: disabled or not configured (optional)."


def main() -> None:
    """Display the environment setup status."""
    print("Future with Ayesha — AI Harness")
    print("Section 1: Environment Setup")
    print("-" * 45)

    if ENV_FILE.is_file():
        print(f"Environment file: found ({ENV_FILE.name})")
    else:
        print(
            "Environment file: .env not found. "
            "Shell environment variables may still be available."
        )

    print(describe_groq_configuration())
    print(describe_langsmith_configuration())

    print("-" * 45)
    print("Setup check complete.")
    print("Note: API credentials have not been tested with a live request.")


if __name__ == "__main__":
    main()