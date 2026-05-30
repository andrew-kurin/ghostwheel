# Ghostwheel Agent

Ghostwheel is a local coding assistant and code-review CLI built with Pydantic AI.
It can talk to either Ollama or a llama.cpp OpenAI-compatible server.

## Install

```bash
uv sync
```

## Run

```bash
uv run ghostwheel
```

In the chat prompt:

- Ask questions about the current repository.
- Use `/review path/to/file.py` to run a focused code review.
- Use `/clear` to reset conversation history.
- Use `/quit` to exit.

## Configuration

Ghostwheel is configured with environment variables prefixed with `GHOSTWHEEL_`.
It also loads a local `.env` file from the working directory. Copy the example file
and edit it for your local model server:

```bash
cp .env.example .env
```

### Ollama

Start Ollama, then use the default provider settings:

```env
GHOSTWHEEL_MODEL_PROVIDER=ollama
GHOSTWHEEL_MODEL=gemma4:26b
GHOSTWHEEL_MODEL_BASE_URL=http://localhost:11434/v1
```

### llama.cpp

Start llama.cpp's server with an OpenAI-compatible endpoint, for example:

```bash
llama-server --hf-repo ggml-org/gemma-4-26B-A4B-it-GGUF --hf-file '*Q4_K_M.gguf' --ctx-size 16384
```

Then configure Ghostwheel:

```env
GHOSTWHEEL_MODEL_PROVIDER=llama-cpp
GHOSTWHEEL_MODEL=ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M
GHOSTWHEEL_MODEL_BASE_URL=http://localhost:8080/v1
```

## Formatter model

Ghostwheel uses a second formatter agent to convert prose code reviews into a
structured table. By default it reuses the main model/provider. You can override
it separately:

```env
GHOSTWHEEL_FORMATTER_PROVIDER=ollama
GHOSTWHEEL_FORMATTER_MODEL=gemma4:26b
GHOSTWHEEL_FORMATTER_BASE_URL=http://localhost:11434/v1
```

## Other settings

```env
GHOSTWHEEL_MAX_OUTPUT_BYTES=100000
GHOSTWHEEL_FORMATTER_RETRIES=5
```
