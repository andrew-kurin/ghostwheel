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

Ghostwheel uses a persistent full-screen chat interface when stdin and stdout
are terminals, and automatically falls back to a plain streaming interface for
pipes and redirected output. Select a mode explicitly with `--ui interactive`
or `--ui plain`.

Interactive input supports command and review-path completion plus ↑/↓ prompt
history. Shift+Enter inserts a newline. The composer grows upward for multiline
or wrapped prompts and returns to its compact height after submission. Input
history is stored in
`$XDG_STATE_HOME/ghostwheel/input-history` (or
`~/.local/state/ghostwheel/input-history`); use `--no-history` to keep it only in
memory or `--history-file PATH` to choose another location. History contains
prompts in plain text, so disable it when prompts may contain secrets.

Vim-style prompt editing is enabled by default. It starts each prompt in Insert
mode; Escape switches to Normal mode, and `i`, `a`, `I`, `A`, `o`, or `O` return
to Insert mode. The compact `I` or `N` beside `You` shows the current mode. Run
`/help` for the available motions and editing commands, or use `--no-vim` to
restore the standard prompt editor.

In the chat prompt:

- Ask questions about the current repository.
- Use `/review path/to/file.py` to run a focused code review.
- Use `/clear` to reset conversation history.
- Use `/retry` to repeat the previous chat or review.
- Use `/model`, `/tools`, or `/help` for runtime information.
- Use `/quit` to exit.

During an active turn, Ctrl+C cancels that turn and returns to the composer. At
the composer, Ctrl+O toggles expanded thinking traces and tool-call results,
including details from existing turns in the visible transcript. Detail widgets
are collapsed by default and are removed from the layout again when collapsed.
Assistant replies render as Markdown in interactive mode; tool calls remain
visible as compact status rows with completion time. Review findings switch to
stacked cards on narrow terminals.

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

GHOSTWHEEL_HISTORY_CONTEXT_WINDOW_TOKENS=16384
GHOSTWHEEL_COMPACTION_ENABLED=true
GHOSTWHEEL_COMPACTION_RESERVE_TOKENS=4096
GHOSTWHEEL_COMPACTION_KEEP_RECENT_TOKENS=4096
GHOSTWHEEL_COMPACTION_SUMMARY_TOKENS=2048
```

Filesystem tools share one canonical workspace policy and output budget. They open
paths relative to allowed-root descriptors and do not traverse symlinks. Chat
history uses rolling summaries rather than a turn-count limit. Compaction starts
when context usage exceeds `context window - reserve tokens`: older messages are
summarized by a dedicated tool-free model call and the newest
`keep recent tokens` remain verbatim. A cut may split a large tool-heavy turn,
but tool calls and their results stay together. Each later compaction folds the
previous summary into the next one.

The context-window value must match the active model server setting. Ghostwheel
uses usage reported by the provider when available; otherwise it estimates with
`tiktoken` and marks the terminal value with `~`. Provider measurements also
calibrate otherwise invisible system-instruction and tool-schema overhead after
a summary is created; that static overhead remains visible after `/clear`. The
4,096-token default reserve leaves response capacity;
the 4,096-token recent target and 2,048-token summary cap leave working room in
the default 16K window. Oversized summarizer inputs are processed as bounded
rolling chunks. Review transcripts do not enter chat history. Set
`GHOSTWHEEL_COMPACTION_ENABLED=false` to disable automatic summaries; no hidden
turn-count limit is applied.

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
