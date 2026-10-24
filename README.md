# Section 1 — LangChain Foundations

**Future with Ayesha — AI Harness** · Prompt templates, chat models, chains, local models, and tracing.

## What you'll learn

- How to keep API keys out of your code using a `.env` file.
- The three building blocks used throughout this project: **PromptTemplate**, **chat model**, and **chain** (`prompt | llm`).
- How to switch between a cloud model using Groq and a local model using Ollama.
- What an LLM call returns through the `AIMessage` object, including generated text, token usage, and metadata.
- How to trace model calls in LangSmith without adding tracing code to every script.

## Files in this section

| File | What it demonstrates | Requirements |
|---|---|---|
| `01_hello_world_env_setup.py` | Loading environment variables safely and displaying a masked API key | `.env` configuration |
| `02_summary_chain_openai.py` | Building a summary chain with `PromptTemplate` and a Groq chat model | `GROQ_API_KEY` |
| `03_summary_chain_ollama.py` | Running the same type of chain with a local Ollama model | Ollama and `gemma3:270m` |
| `04_inspect_the_response_object.py` | Inspecting the returned `AIMessage`, including content and metadata | `GROQ_API_KEY`, or Ollama with `--local` |
| `sample_data.py` | Biography text used by the example scripts | None |

## Setup

Create and activate a virtual environment, then install the project's dependencies.

If the project contains `pyproject.toml` and `uv.lock`, run:

```bash
uv sync
cp .env.example .env
```

Otherwise, install the dependencies listed in your project's requirements file and create `.env` from `.env.example` if that template is available.

Add your Groq API key to `.env`:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=section-01-langchain-foundations

OLLAMA_MODEL=gemma3:270m
```

Replace the placeholder values with your own credentials. Never commit `.env` or share your API keys.

## Run the examples

```bash
uv run python 01_hello_world_env_setup.py
uv run python 02_summary_chain_openai.py
uv run python 04_inspect_the_response_object.py
```

The filenames are retained from the original exercises, even though the summary and inspection scripts use Groq as the cloud provider in this setup.

### Run with Ollama

Install Ollama from [ollama.com/download](https://ollama.com/download), start the application, and download the model:

```bash
ollama pull gemma3:270m
```

Then run:

```bash
uv run python 03_summary_chain_ollama.py
uv run python 04_inspect_the_response_object.py --local
```

The local examples require Ollama to be installed and running. The configured model must also be available locally.

## Expected output

The summary example produces three sections:

1. **Headline:** A short description of the biography.
2. **Summary for a 12-year-old:** Three simple bullet points explaining the person's life and achievements.
3. **Interview question:** One question based on the biography.

The exact wording can change between runs. The prompt defines the expected structure, while the model generates the actual text.

## How it works

```text
Biography text
      |
      v
PromptTemplate
Fills the {information} variable
      |
      v
ChatGroq / ChatOllama
Sends the formatted prompt to the selected model
      |
      v
AIMessage
Generated content, token usage, and response metadata
```

- `load_dotenv()` loads environment variables from `.env`.
- `prompt | llm` combines the prompt and model into a `RunnableSequence`.
- `chain.invoke({...})` executes the chain with the supplied input.
- `response.content` provides the model's response content. Its exact structure can vary by model and LangChain version, so inspect the returned object rather than assuming every field is a plain string.
- LangSmith can capture traces when tracing is enabled and the required credentials are configured.

## LangSmith tracing

Add the following configuration to `.env`:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=section-01-langchain-foundations
```

Run one of the scripts and visit [LangSmith](https://smith.langchain.com) to inspect the resulting traces.

Depending on the chain and model, traces can help you examine the prompt, model response, execution time, token usage, and other available metadata.

## Formatting (optional)

```bash
uv run black .
uv run isort .
```

These commands require `black` and `isort` to be included in the project's development dependencies.

## Troubleshooting

| Problem | What to check |
|---|---|
| `GROQ_API_KEY is missing` | Confirm that `.env` exists and contains `GROQ_API_KEY`. |
| Authentication error or HTTP 401 | Check that the Groq API key is valid and has not been revoked. |
| Model not found | Check that `GROQ_MODEL` names a model currently available to your Groq account. |
| Ollama connection refused | Install and start Ollama before running the local scripts. |
| Ollama model not found | Run `ollama pull gemma3:270m` or pull the model configured in `OLLAMA_MODEL`. |
| LangSmith traces are missing | Check `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT`. |
| `.env` appears in Git status | Add `.env` to `.gitignore`. If it is already tracked, remove it from Git's index before committing. |

## What I learned

This section establishes the foundation for the later exercises: loading configuration securely, creating prompt templates, connecting chat models, composing chains, examining responses, and tracing model execution.


