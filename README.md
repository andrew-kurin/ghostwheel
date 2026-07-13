# Ghostwheel

See [ARCHITECTURE.md](ARCHITECTURE.md) for module ownership and dependency
direction.

Ghostwheel is a local-first coding assistant and code-review CLI built with
Pydantic AI. It connects to either Ollama or a llama.cpp OpenAI-compatible
server.

Chat and review models can request any tools registered for their profile.
Ghostwheel gives its shell tool unrestricted access to the environment in which
the CLI is running, without a command-approval gate. Run it inside a sandbox or
worktree whose files and processes you are willing to expose. Tool profiles
control which capabilities are registered; they are not a shell-command sandbox.
Configured model endpoints receive prompts and tool results, which can include
source code, so use only endpoints you trust.

Ghostwheel currently requires a POSIX environment (macOS or Linux), including
`bash`, `openat`, and `O_NOFOLLOW`. It fails fast where the secure workspace
adapter cannot provide those guarantees.

## Install

Requirements:

- Python 3.14 or newer
- [`uv`](https://docs.astral.sh/uv/)
- A supported model server: [Ollama](https://ollama.com/) or
  [llama.cpp](https://github.com/ggml-org/llama.cpp)

```bash
git clone https://github.com/andrew-kurin/ghostwheel.git
cd ghostwheel
uv sync
```

## Run

```bash
uv run ghostwheel
```

Ghostwheel runs inline in the terminal's primary screen buffer. Completed turns
remain in native scrollback, and the terminal emulator owns text selection,
copy/paste, search, wheel scrolling, and its scrollbar. When stdout is
redirected, Ghostwheel writes a line-oriented stream without terminal control
sequences. Redirecting stdin alone also switches input to a line-oriented
stream, but output can retain terminal styling when stdout is still a terminal.
Run `uv run ghostwheel --help` for all command-line options.

The interactive prompt-toolkit composer supports multiline editing, command and
review-path completion, and prompt history. Enter submits and Shift+Enter
inserts a newline when the terminal sends either xterm's legacy
modifyOtherKeys sequence (`\x1b[27;2;13~`) or the CSI-u/Kitty sequence
(`\x1b[13;2u`). Terminals that send ordinary Enter for this shortcut need to
map Shift+Enter to one of those sequences. Ctrl+J has no action.

When the interactive composer is active, prompt history is stored in
`$XDG_STATE_HOME/ghostwheel/input-history` (or
`~/.local/state/ghostwheel/input-history`); use `--no-history` to keep it only in
memory or `--history-file PATH` to choose another location. These history
options do not apply to line-oriented input. History contains prompts in plain
text, so disable it when prompts may contain secrets. If the history file
cannot be read or written, Ghostwheel warns once and continues with in-memory
history. Model conversation history and rolling summaries are memory-only and
disappear when Ghostwheel exits.

Vim-style prompt editing is enabled by default through prompt-toolkit. Each
prompt starts in Insert mode; Escape switches to Normal mode, and the
ruled status line at the bottom shows context usage and the current `I`, `N`, or
`R` editing mode. Use `--no-vim` for Emacs-style editing.

In the chat prompt:

- Ask questions about the current repository.
- Use `/review path/to/file.py` to run a focused code review.
- Use `/clear` to reset conversation history.
- Use `/retry` to repeat the previous chat or review.
- Use `/model` for both active models, `/tools` for tool profiles, or `/help`.
- Use `/quit` to exit.

During an active turn, Esc cancels and returns to the composer. Ctrl+C clears
the current prompt, and Ctrl+D exits unconditionally. Other text entered while a
turn is running is discarded so it cannot corrupt the live preview or leak into
the next prompt.
Active turns use a bounded live preview; completed assistant replies are
committed to scrollback as Markdown. Tool calls remain visible as compact status
rows with completion time, and review findings switch to stacked cards on narrow
terminals.

## Configuration

Ghostwheel is configured with environment variables prefixed with `GHOSTWHEEL_`.
It also loads a local `.env` file from the working directory. Copy the example file
and edit it for your local model server:

```bash
cp .env.example .env
```

Inspect a workspace's `.env` before running Ghostwheel. It can redirect model
traffic, change the registered tool profiles, and enable observability.

### Ollama

Pull the default model and configure Ollama with the same context window used by
Ghostwheel. For a CLI-served Ollama instance, for example:

```bash
ollama pull gemma4:26b
OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

The Ollama app setting or a Modelfile with `PARAMETER num_ctx 16384` are
equivalent ways to set the server context. Then use the default provider
settings:

```env
GHOSTWHEEL_MODEL_PROVIDER=ollama
GHOSTWHEEL_MODEL=gemma4:26b
GHOSTWHEEL_MODEL_BASE_URL=http://localhost:11434/v1
```

### llama.cpp

Start llama.cpp's server with an OpenAI-compatible endpoint, for example:

```bash
llama-server \
  --hf-repo ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M \
  --ctx-size 16384 \
  --jinja
```

Then configure Ghostwheel:

```env
GHOSTWHEEL_MODEL_PROVIDER=llama-cpp
GHOSTWHEEL_MODEL=ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M
GHOSTWHEEL_MODEL_BASE_URL=http://localhost:8080/v1
```

## Review model

Reviews run with fresh model-message context and request a structured result
directly from the review model. This is not process or workspace isolation:
reviews use the same workspace and their configured tool profile, which defaults
to `full`. By default the review model reuses the main model/provider; it can be
overridden independently:

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
GHOSTWHEEL_MAX_READ_LINES=200
GHOSTWHEEL_MAX_READ_SCAN_BYTES=5000000
GHOSTWHEEL_MAX_ENTRIES=200
GHOSTWHEEL_MAX_DIRECTORY_SCAN_ENTRIES=10000
GHOSTWHEEL_MAX_MATCHES=200
GHOSTWHEEL_BASH_TIMEOUT_SECONDS=30
GHOSTWHEEL_MAX_SEARCH_FILE_BYTES=5000000
GHOSTWHEEL_MAX_SEARCH_TOTAL_BYTES=50000000
GHOSTWHEEL_MAX_SEARCH_FILES=10000
GHOSTWHEEL_SEARCH_TIMEOUT_SECONDS=5.0
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
paths relative to allowed-root descriptors and do not traverse symlinks. The
`read-only` profile registers `read`, `ls`, and `grep`; `shell-only` registers
unrestricted `bash`; and `full` registers all four. Chat and review profiles are
configured independently.

`read` returns UTF-8 text as compact `LINE:TEXT` rows, with at most
`GHOSTWHEEL_MAX_READ_LINES` rows per page. Callers can request a smaller limit,
jump to a 1-based starting line, or continue sequentially with the returned
cursor. Cursors seek directly to the next line and are rejected if the path or
opened file changes. The header reports the returned range, byte size,
completeness, and whether a page, output, or long-line limit was reached. Output
contains only complete rows; exceptionally long lines end with an explicit
truncation marker. NUL-containing and invalid UTF-8 files are rejected instead
of being returned as replacement-character noise when encountered. To keep
random line jumps and pathological single lines bounded,
`GHOSTWHEEL_MAX_READ_SCAN_BYTES` caps bytes inspected by each call.

`ls` returns JSON-escaped rows in sorted relative-path order, with `f`, `d`, `l`,
and `o` markers for files, directories, symlinks, and other filesystem entries.
It lists one level without file sizes by default; callers can request a depth
from 1–3, a case-sensitive glob, exact file sizes, common dependency/cache
directories, a smaller page limit, or the next page using the returned cursor.
The configured scan, entry, and output limits remain hard ceilings shared across
the requested tree. Listings report why they are incomplete and omit a
continuation cursor after scan or entry errors; a scan-limited subset is sorted
but cannot guarantee complete deterministic membership.

`grep` searches UTF-8 text one line at a time using the Python `regex` package,
or exact text when `literal=true`. It returns each matching line once at the
first match, grouped under an ASCII JSON-escaped file path as
`LINE:COLUMN[~] JSON_SNIPPET`; `~` marks a match-centered snippet elided from a
long line. Directory searches accept a recursive relative `file_glob`, are case
sensitive by default, and omit hidden entries plus `.git`, `.venv`,
`__pycache__`, and `node_modules` unless requested. Callers can also select
case-insensitive or literal matching, include hidden/noise paths, set a smaller
page limit, and continue an unchanged result set with its cursor.
Continuations are stateless: each page reruns the bounded search and verifies
that its result snapshot is unchanged before applying the cursor.

The `grep` header reports scanned entries, searched and skipped files, bytes
inspected, completeness, and explicit reasons for incomplete results. Binary,
invalid UTF-8, oversized, and inaccessible files are skipped rather than searched
lossily. `GHOSTWHEEL_MAX_SEARCH_FILE_BYTES` caps each file;
`GHOSTWHEEL_MAX_SEARCH_TOTAL_BYTES` caps cumulative bytes per call;
`GHOSTWHEEL_MAX_SEARCH_FILES` caps eligible files; and
`GHOSTWHEEL_SEARCH_TIMEOUT_SECONDS` sets a cooperative deadline checked across
the complete search.
`GHOSTWHEEL_REGEX_TIMEOUT_SECONDS` separately bounds each line-oriented regex
operation. `GHOSTWHEEL_MAX_MATCHES` caps collected matching lines and the maximum
page size, while `GHOSTWHEEL_MAX_DIRECTORY_SCAN_ENTRIES` bounds traversal shared
with `ls`.

Chat history uses rolling summaries rather than a turn-count limit. Before each
new chat request, Ghostwheel projects context usage including the pending prompt.
When that projection exceeds `context window - reserve tokens`, older messages
are summarized with the active chat model in a separate, tool-free call. The
policy aims to preserve roughly `keep recent tokens` verbatim, but safe tool-pair
boundaries and the space needed for the pending request can make the retained
suffix larger or smaller. Cuts do not separate a tool call from its result, but
long serialized content can be truncated and model-generated summaries are
inherently lossy. Each later compaction folds the previous summary into the next
one, adding a model call and its associated latency.

The context-window value must match the active model server setting. Ghostwheel
uses usage reported by the provider when available; otherwise it estimates with
`tiktoken`'s model-independent `o200k_base` encoding and marks the terminal value
with `~`. This proxy can differ from the model server's tokenizer, and its first
use may need network access to populate tiktoken's encoding cache. When a
successful chat response includes provider usage, Ghostwheel uses it to calibrate
otherwise invisible system-instruction, tool-schema, chat-template, and tokenizer
overhead. That calibrated overhead remains visible after `/clear`. The prompt's
bottom status shows `USED/WINDOW`; `~` marks an estimate and `· off` means
automatic compaction is disabled. The value can decrease when `~` disappears:
that means a provider measurement replaced the conservative tokenizer proxy.
The 4,096-token default reserve leaves response capacity;
the 4,096-token recent target and 2,048-token summary cap leave working room in
the default 16K window. Oversized summarizer inputs are processed as bounded
rolling chunks. Review transcripts do not enter chat history. Set
`GHOSTWHEEL_COMPACTION_ENABLED=false` to disable automatic summaries; no hidden
turn-count limit is applied, but Ghostwheel will no longer trim growing context,
so the provider may reject or truncate oversized requests.

`GHOSTWHEEL_MAX_OUTPUT_BYTES` caps the complete compact text that `read`, `ls`,
and `grep` send to the model. For structured process results it caps retained
variable payload; its small result envelope is additional.

## Observability

Logfire instrumentation is disabled by default. It must be explicitly enabled,
and prompt/tool content has a separate opt-in:

```env
GHOSTWHEEL_OBSERVABILITY_ENABLED=false
GHOSTWHEEL_OBSERVABILITY_INCLUDE_CONTENT=false
GHOSTWHEEL_OBSERVABILITY_SEND_TO_LOGFIRE=if-token-present
```
