# Ghostwheel Agent

Ghostwheel is a local coding assistant and code-review CLI built with Pydantic AI.
It can talk to either Ollama or a llama.cpp OpenAI-compatible server.

Ghostwheel gives its shell tool unrestricted access to the environment in which
the CLI is running. Run it inside a sandbox or worktree whose files and processes
you are willing to expose. Tool profiles control which capabilities are registered;
they are not a shell-command sandbox.

Ghostwheel currently requires a POSIX environment (macOS or Linux), including
`bash`, `openat`, and `O_NOFOLLOW`. It fails fast where the secure workspace
adapter cannot provide those guarantees.

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

## Review model

Reviews run in fresh context and request a structured result directly from the
review model. By default the review model reuses the main model/provider; it can
be overridden independently:

```env
GHOSTWHEEL_REVIEW_PROVIDER=ollama
GHOSTWHEEL_REVIEW_MODEL=gemma4:26b
GHOSTWHEEL_REVIEW_BASE_URL=http://localhost:11434/v1
GHOSTWHEEL_REVIEW_RETRIES=5
GHOSTWHEEL_REVIEW_RAW_FALLBACK=true
```

When a provider rejects or repeatedly fails structured output, the fallback runs
the review as prose and transcribes that prose into the same validated schema. It
shows raw prose only if transcription also fails; network and tool failures are
reported immediately rather than rerunning the review.

The former `GHOSTWHEEL_FORMATTER_PROVIDER`, `GHOSTWHEEL_FORMATTER_MODEL`, and
`GHOSTWHEEL_FORMATTER_BASE_URL` variables are used as one compatibility tier only
when no review-model override is set. Review retries independently fall back to
`GHOSTWHEEL_FORMATTER_RETRIES`.

## Other settings

```env
GHOSTWHEEL_MAX_OUTPUT_BYTES=100000
GHOSTWHEEL_MAX_ENTRIES=200
GHOSTWHEEL_MAX_DIRECTORY_SCAN_ENTRIES=10000
GHOSTWHEEL_MAX_MATCHES=200
GHOSTWHEEL_BASH_TIMEOUT_SECONDS=30
GHOSTWHEEL_MAX_SEARCH_FILE_BYTES=5000000
GHOSTWHEEL_MAX_SEARCH_FILES=10000
GHOSTWHEEL_REGEX_TIMEOUT_SECONDS=0.05

# full, read-only, or shell-only
GHOSTWHEEL_TOOL_PROFILE=full
GHOSTWHEEL_REVIEW_TOOL_PROFILE=full

GHOSTWHEEL_HISTORY_MAX_TURNS=20
GHOSTWHEEL_HISTORY_MAX_MESSAGES=200
GHOSTWHEEL_HISTORY_MAX_BYTES=400000
GHOSTWHEEL_HISTORY_RESPONSE_RESERVE_BYTES=50000
```

Filesystem tools share one canonical workspace policy and output budget. They open
paths relative to allowed-root descriptors and do not traverse symlinks. Chat
history is retained as whole turns and compacted when any configured history limit
is reached; review transcripts do not enter chat history.

`GHOSTWHEEL_MAX_OUTPUT_BYTES` limits retained variable payload (file content,
matches, and process streams); the small structured result envelope is additional.

## Observability

Logfire instrumentation is disabled by default. It must be explicitly enabled,
and prompt/tool content has a separate opt-in:

```env
GHOSTWHEEL_OBSERVABILITY_ENABLED=false
GHOSTWHEEL_OBSERVABILITY_INCLUDE_CONTENT=false
GHOSTWHEEL_OBSERVABILITY_SEND_TO_LOGFIRE=if-token-present
```
