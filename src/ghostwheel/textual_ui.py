"""Persistent full-screen terminal UI for interactive Ghostwheel sessions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

from rich.console import Group, RenderableType
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.segment import Segment
from rich.style import Style as RichStyle
from rich.text import Text
from textual import events
from textual.app import App, AutopilotCallbackType, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.styles import RulesMap
from textual.drivers.linux_driver import LinuxDriver
from textual.selection import Selection
from textual.strip import Strip
from textual.style import Style as TextualStyle
from textual.visual import RenderOptions, RichVisual, Visual
from textual.worker import Worker, WorkerCancelled, WorkerFailed
from textual.widgets import Static, TextArea

from ghostwheel.app_info import AppInfo
from ghostwheel.cancellation import TurnCancellation
from ghostwheel.controller import ReviewPort, SessionPort, run_command_loop
from ghostwheel.events import (
    AgentEvent,
    TextOutput,
    ThinkingOutput,
    ToolFailed,
    ToolFinished,
    ToolStarted,
)
from ghostwheel.input_ui import InputHistory
from ghostwheel.presentation import (
    ToolActivity,
    TurnState,
    duration,
    failure_presentation,
    format_token_count,
    preview,
    primary_argument,
)
from ghostwheel.rendering import review_renderables
from ghostwheel.review import RawReview, ReviewFailed, ReviewOutcome, StructuredReview
from ghostwheel.runtime_contracts import (
    TurnFailed,
    TurnNoResult,
    TurnOutcome,
    TurnSucceeded,
)
from ghostwheel.textual_composer import Composer, VimMode

MODIFY_OTHER_KEYS_ENABLE = "\x1b[>4;2m"
MODIFY_OTHER_KEYS_RESET = "\x1b[>4;m"
COMPOSER_MIN_HEIGHT = 2
COMPOSER_BORDER_HEIGHT = 1
TRANSCRIPT_MIN_HEIGHT = 4


class GhostwheelTerminalDriver(LinuxDriver):
    """Add xterm extended-key fallback to Textual's Kitty negotiation."""

    def start_application_mode(self) -> None:
        super().start_application_mode()
        if self._writer_thread is not None:
            self.write(MODIFY_OTHER_KEYS_ENABLE)

    def stop_application_mode(self) -> None:
        if self._writer_thread is not None:
            self.write(MODIFY_OTHER_KEYS_RESET)
        super().stop_application_mode()


class QueueInputReader:
    """Feed submitted composer text into the existing asynchronous CLI loop."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def read(self) -> str:
        return await self._queue.get()

    def submit(self, value: str) -> None:
        self._queue.put_nowait(value)


class _SelectableRichVisual(Visual):
    """Add Textual selection offsets and highlighting to a Rich visual."""

    def __init__(self, visual: RichVisual) -> None:
        self._visual = visual
        self.plain_lines: list[str] = []

    def get_optimal_width(self, rules: RulesMap, container_width: int) -> int:
        return self._visual.get_optimal_width(rules, container_width)

    def get_height(self, rules: RulesMap, width: int) -> int:
        return self._visual.get_height(rules, width)

    def render_strips(
        self,
        width: int,
        height: int | None,
        style: TextualStyle,
        options: RenderOptions,
    ) -> list[Strip]:
        strips = self._visual.render_strips(width, height, style, options)
        self.plain_lines = [strip.text.rstrip() for strip in strips]
        return [
            self._decorate_strip(strip, line_number, options)
            for line_number, strip in enumerate(strips)
        ]

    @staticmethod
    def _decorate_strip(
        strip: Strip,
        line_number: int,
        options: RenderOptions,
    ) -> Strip:
        selection_span = (
            options.selection.get_span(line_number)
            if options.selection is not None
            else None
        )
        output: list[Segment] = []
        line_offset = 0

        for text, segment_style, control in strip:
            if control:
                output.append(Segment(text, segment_style, control))
                continue

            segment_start = line_offset
            segment_end = segment_start + len(text)
            cuts = {0, len(text)}
            if selection_span is not None:
                selection_start, selection_end = selection_span
                if segment_start < selection_start < segment_end:
                    cuts.add(selection_start - segment_start)
                if selection_end >= 0 and segment_start < selection_end < segment_end:
                    cuts.add(selection_end - segment_start)

            ordered_cuts = sorted(cuts)
            for start, end in zip(ordered_cuts, ordered_cuts[1:]):
                piece = text[start:end]
                piece_offset = segment_start + start
                rich_style = segment_style or RichStyle.null()
                if selection_span is not None and options.selection_style is not None:
                    selection_start, selection_end = selection_span
                    if piece_offset >= selection_start and (
                        selection_end < 0 or piece_offset < selection_end
                    ):
                        rich_style = (
                            TextualStyle.from_rich_style(rich_style)
                            + options.selection_style
                        ).rich_style
                rich_style += RichStyle.from_meta(
                    {"offset": (piece_offset, line_number)}
                )
                output.append(Segment(piece, rich_style))
            line_offset = segment_end

        return Strip(output, strip.cell_length)


class SelectableStatic(Static):
    """Static content that remains selectable when backed by Rich renderables."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._selectable_source: RichVisual | None = None
        self._selectable_visual: _SelectableRichVisual | None = None

    def render(self) -> Visual:
        visual = super().render()
        if not isinstance(visual, RichVisual):
            return visual
        if visual is not self._selectable_source:
            self._selectable_source = visual
            self._selectable_visual = _SelectableRichVisual(visual)
        assert self._selectable_visual is not None
        return self._selectable_visual

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        visual = self._render()
        if isinstance(visual, _SelectableRichVisual) and visual.plain_lines:
            return selection.extract("\n".join(visual.plain_lines)), "\n"
        return super().get_selection(selection)


class TurnView(Vertical):
    """One assistant turn with retained, dynamically hideable details."""

    def __init__(self, label: str, *, details_expanded: bool = False) -> None:
        self.status = SelectableStatic(Text(label, style="dim"), classes="turn-status")
        self.tool_summaries = SelectableStatic(classes="tool-summaries")
        self.tool_details = SelectableStatic(classes="turn-details tool-details")
        self.thinking_detail = SelectableStatic(classes="turn-details thinking-detail")
        self.heading = SelectableStatic(
            Text("Ghostwheel", style="bold magenta"),
            classes="assistant-heading",
        )
        self.answer = SelectableStatic(classes="assistant-answer")
        self.outcome = SelectableStatic(classes="turn-outcome")
        super().__init__(
            self.status,
            self.tool_summaries,
            self.tool_details,
            self.thinking_detail,
            self.heading,
            self.answer,
            self.outcome,
            classes="turn",
        )
        self.details_expanded = details_expanded
        self.state = TurnState(label)
        self.tool_summaries.display = False
        self.tool_details.display = False
        self.thinking_detail.display = False
        self.heading.display = False
        self.answer.display = False
        self.outcome.display = False

    async def apply_event(self, event: AgentEvent) -> None:
        self.state.apply(event)
        if isinstance(event, ThinkingOutput):
            self.status.update(Text(self.state.status, style="dim"))
            self._render_thinking()
        elif isinstance(event, TextOutput):
            self.status.update(Text(self.state.status, style="dim"))
            self.heading.display = True
            self.answer.display = True
            self.answer.update(RichMarkdown(self.state.answer))
        elif isinstance(event, ToolStarted):
            self.status.update(Text(self.state.status, style="dim"))
            self._render_tools()
        elif isinstance(event, ToolFinished):
            self.status.update(Text(self.state.status, style="dim"))
            self._render_tools()
        elif isinstance(event, ToolFailed):
            self.status.update(Text(self.state.status, style="red"))
            self._render_tools()

    def set_details_expanded(self, expanded: bool) -> None:
        self.details_expanded = expanded
        self._render_thinking()
        self._render_tools()

    def finish_turn(self, outcome: TurnOutcome) -> None:
        self.status.display = False
        if isinstance(outcome, TurnSucceeded):
            self.state.answer = outcome.output
            self.heading.display = True
            self.answer.display = True
            self.answer.update(RichMarkdown(outcome.output))
        elif isinstance(outcome, TurnNoResult):
            self._show_outcome(Text(outcome.message, style="yellow"))
        elif isinstance(outcome, TurnFailed):
            self._show_outcome(_turn_failure(outcome))

    def finish_review(self, outcome: ReviewOutcome, *, width: int) -> None:
        self.status.display = False
        if isinstance(outcome, StructuredReview):
            renderables: list[RenderableType] = []
            if outcome.used_fallback:
                renderables.append(
                    Text(
                        "Structured-output fallback was used for this review.",
                        style="dim",
                    )
                )
            renderables.extend(review_renderables(outcome.review, width=width))
            self._show_outcome(Group(*renderables))
        elif isinstance(outcome, RawReview):
            body = Text()
            body.append("Couldn't produce a structured review.\n", style="yellow")
            body.append("Reason: ", style="dim")
            body.append(outcome.structured_failure, style="dim")
            body.append("\n\nShowing the raw review instead:\n\n", style="bold")
            body.append(outcome.prose)
            self._show_outcome(
                Panel(body, title="Structured Review Failed", border_style="yellow")
            )
        elif isinstance(outcome, ReviewFailed):
            body = Text(outcome.message)
            body.append(
                "\n\nCheck the review model configuration, then use /retry.",
                style="dim",
            )
            self._show_outcome(Panel(body, title="Review Failed", border_style="red"))

    def cancel(self) -> None:
        self.status.display = False
        self._show_outcome(Text("Turn cancelled.", style="yellow"))

    def _show_outcome(self, renderable: RenderableType) -> None:
        self.outcome.update(renderable)
        self.outcome.display = True

    def _render_thinking(self) -> None:
        self.thinking_detail.update(
            Panel(
                Text(self.state.thinking, style="dim"),
                title=Text("Thinking", style="dim"),
                border_style="dim",
                padding=(0, 1),
            )
        )
        self.thinking_detail.display = (
            bool(self.state.thinking) and self.details_expanded
        )

    def _render_tools(self) -> None:
        summaries = [_tool_summary(tool) for tool in self.state.tools]
        details = [_tool_detail(tool) for tool in self.state.tools]
        self.tool_summaries.update(Group(*summaries))
        self.tool_details.update(Group(*details))
        self.tool_summaries.display = bool(summaries)
        self.tool_details.display = bool(details) and self.details_expanded

    @property
    def _answer(self) -> str:
        """Compatibility view of the reducer-backed answer."""

        return self.state.answer

    @property
    def _thinking(self) -> str:
        """Compatibility view of the reducer-backed thinking text."""

        return self.state.thinking

    @property
    def _tools(self) -> list[ToolActivity]:
        """Compatibility view of reducer-backed tool activity."""

        return self.state.tools


class TextualPresenter:
    """Presenter adapter used by the existing command loop inside Textual."""

    def __init__(self, app: GhostwheelApp, app_info: AppInfo) -> None:
        self.app = app
        self.app_info = app_info
        self.current_turn: TurnView | None = None

    async def handle_event(self, event: AgentEvent) -> None:
        turn = self.current_turn
        if turn is None:
            turn = self._start_turn("Thinking…")
        follow = self.app.transcript.is_vertical_scroll_end
        await turn.apply_event(event)
        self.app.follow_output(follow)

    def welcome(self) -> None:
        # The persistent header already carries the application identity.
        return

    def goodbye(self) -> None:
        self.app.exit()

    def help(self) -> None:
        self.app.add_renderable(
            _help_panel(vim_mode=self.app.vim_mode),
            classes="system-message",
        )

    def model_info(self) -> None:
        self.app.add_renderable(
            Text.assemble(
                Text("Model  ", style="bold"),
                Text(
                    f"{self.app_info.provider}/{self.app_info.model}",
                    style="cyan",
                ),
            ),
            classes="system-message",
        )

    def tools_info(self) -> None:
        body = Text.assemble(
            Text("Tool profile  ", style="bold"),
            Text(self.app_info.tool_profile, style="yellow"),
        )
        if self.app_info.tool_profile == "full":
            body.append("\nShell commands run with unrestricted environment access.")
        self.app.add_renderable(
            Panel(body, title="Tools", border_style="yellow"),
            classes="system-message",
        )

    def unknown_command(self, command: str, suggestion: str | None = None) -> None:
        message = Text.assemble(
            Text("Unknown command: ", style="yellow"),
            Text(command),
        )
        if suggestion:
            message.append(f"\nDid you mean {suggestion}?", style="dim")
        message.append("\nType /help to list commands.", style="dim")
        self.app.add_renderable(message, classes="system-message")

    def retry_unavailable(self) -> None:
        self.app.add_renderable(
            Text("Nothing to retry yet.", style="yellow"),
            classes="system-message",
        )

    def history_cleared(self) -> None:
        self.app.update_context()
        self.app.add_renderable(
            Text("Conversation history cleared.", style="dim"),
            classes="system-message",
        )

    def history_compacted(self, before_tokens: int, after_tokens: int) -> None:
        self.app.add_renderable(
            Text(
                "Context compacted: "
                f"{format_token_count(before_tokens)} → "
                f"~{format_token_count(after_tokens)}.",
                style="dim",
            ),
            classes="system-message",
        )

    def turn_started(self, label: str = "Thinking…") -> None:
        self._start_turn(label)

    def turn_cancelled(self) -> None:
        if self.current_turn is not None:
            self.current_turn.cancel()
        self.current_turn = None
        self.app.follow_output(True)

    def turn_outcome(self, outcome: TurnOutcome) -> None:
        follow = self.app.transcript.is_vertical_scroll_end
        if self.current_turn is None:
            self._start_turn("Thinking…")
        assert self.current_turn is not None
        self.current_turn.finish_turn(outcome)
        self.current_turn = None
        self.app.update_context()
        self.app.follow_output(follow)

    def review_outcome(self, outcome: ReviewOutcome) -> None:
        follow = self.app.transcript.is_vertical_scroll_end
        if self.current_turn is None:
            self._start_turn("Reviewing…")
        assert self.current_turn is not None
        self.current_turn.finish_review(outcome, width=max(40, self.app.size.width - 4))
        self.current_turn = None
        self.app.follow_output(follow)

    def _start_turn(self, label: str) -> TurnView:
        turn = TurnView(label, details_expanded=self.app.details_expanded)
        self.current_turn = turn
        self.app.add_widget(turn)
        return turn


class GhostwheelApp(App[None]):
    """Full-screen chat UI with a redrawable transcript and composer."""

    TITLE = "Ghostwheel"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("ctrl+o", "toggle_details", show=False, priority=True),
        Binding("super+c", "copy_selection", show=False, priority=True),
        Binding("ctrl+c", "cancel_or_quit", show=False, priority=True),
        Binding("ctrl+q", "quit", show=False, priority=True),
    ]
    CSS = """
    Screen {
        background: $background;
        color: $text;
    }

    #app-header {
        height: 3;
        padding: 0 1;
        background: $background;
        color: $text;
    }

    #transcript {
        height: 1fr;
        padding: 0 1 1 1;
        background: $background;
        scrollbar-size-vertical: 1;
        scrollbar-color: $primary-darken-2;
        scrollbar-background: $background;
    }

    .user-message {
        height: auto;
        margin-top: 1;
    }

    .system-message {
        height: auto;
        margin: 1 0 0 2;
    }

    TurnView {
        height: auto;
        margin-top: 1;
    }

    TurnView > Static {
        height: auto;
    }

    .turn-status {
        color: $text-muted;
    }

    .tool-summaries, .turn-details, .turn-outcome, .assistant-answer {
        margin-left: 2;
    }

    .assistant-heading {
        margin-top: 1;
    }

    #composer-shell {
        height: 2;
        padding: 0 1;
        background: $surface;
        border-top: solid $primary-darken-3;
    }

    #composer {
        width: 1fr;
        height: 1fr;
        border: none;
        padding: 0;
        background: $surface;
        color: $text;
    }

    #composer .text-area--cursor {
        text-style: none;
    }

    #context {
        width: auto;
        height: 1fr;
        content-align: right top;
        color: $text-muted;
        background: $surface;
    }
    """

    def __init__(
        self,
        console: object | SessionPort | None = None,
        session: SessionPort | ReviewPort | None = None,
        reviews: ReviewPort | None = None,
        *,
        app_info: AppInfo,
        history_path: Path | None = None,
        cancellation: TurnCancellation | None = None,
        vim_mode: bool = True,
    ) -> None:
        if reviews is None:
            if console is None or session is None:
                raise TypeError("GhostwheelApp requires session and reviews")
            resolved_console = None
            resolved_session = cast(SessionPort, console)
            resolved_reviews = cast(ReviewPort, session)
        else:
            if session is None:
                raise TypeError("GhostwheelApp requires a session")
            resolved_console = console
            resolved_session = cast(SessionPort, session)
            resolved_reviews = reviews

        super().__init__(driver_class=GhostwheelTerminalDriver, ansi_color=True)
        # ``console`` is retained solely for callers using the original
        # ``(console, session, reviews)`` constructor. Textual owns rendering.
        self._console = resolved_console
        self.session = resolved_session
        self.reviews = resolved_reviews
        self.app_info = app_info
        self.cancellation = cancellation or TurnCancellation()
        self.history = InputHistory(history_path)
        self.input_reader = QueueInputReader()
        self.details_expanded = False
        self.vim_mode = vim_mode
        self._composer_height = COMPOSER_MIN_HEIGHT
        self._right_click_copy_active = False
        self._command_worker: Worker[None] | None = None
        self._command_worker_failure: WorkerFailed | None = None
        self.header = SelectableStatic(_header(app_info), id="app-header")
        self.transcript = VerticalScroll(id="transcript")
        self.composer = Composer(self.history, vim_enabled=vim_mode)
        self.context = Static(id="context")
        self.composer_shell = Horizontal(
            self.composer,
            self.context,
            id="composer-shell",
        )
        self.presenter = TextualPresenter(self, app_info)

    async def run_async(
        self,
        *,
        headless: bool = False,
        inline: bool = False,
        inline_no_clear: bool = False,
        mouse: bool = True,
        size: tuple[int, int] | None = None,
        auto_pilot: AutopilotCallbackType | None = None,
    ) -> None:
        """Run the UI and report command-worker failures after driver cleanup."""

        self._command_worker_failure = None
        await super().run_async(
            headless=headless,
            inline=inline,
            inline_no_clear=inline_no_clear,
            mouse=mouse,
            size=size,
            auto_pilot=auto_pilot,
        )
        if self._command_worker_failure is not None:
            raise self._command_worker_failure

    async def on_event(self, event: events.Event) -> None:
        if isinstance(event, events.MouseEvent) and not event.is_forwarded:
            if self._right_click_copy_active:
                if isinstance(event, events.MouseUp):
                    self._right_click_copy_active = False
                return
            if (
                isinstance(event, events.MouseDown)
                and event.button == 3
                and self._copy_selection()
            ):
                # Consume the complete right-click gesture so Textual doesn't
                # replace the existing selection before the button is released.
                self._right_click_copy_active = True
                return
        await super().on_event(event)

    def compose(self) -> ComposeResult:
        yield self.header
        yield self.transcript
        yield self.composer_shell

    def on_mount(self) -> None:
        self.update_context()
        self.composer.focus()
        self.call_after_refresh(self._resize_composer)
        self._command_worker = self.run_worker(
            run_command_loop(
                self.session,
                self.reviews,
                presenter=self.presenter,
                input_reader=self.input_reader,
                cancellation=self.cancellation,
            ),
            name="command-loop",
            exit_on_error=True,
        )

    async def on_unmount(self) -> None:
        """Wait for an active turn's cancellation cleanup before app shutdown."""

        worker = self._command_worker
        if worker is None:
            return
        try:
            await worker.wait()
        except WorkerCancelled:
            # Textual cancels workers as its message loop stops. ``wait`` only
            # raises after the worker task, including its cleanup, has finished.
            pass
        except WorkerFailed as error:
            # Raising from Unmount prevents Textual's shutdown routine from
            # closing its driver. Defer the same failure until ``run_async``
            # has completed the rest of terminal teardown.
            self._command_worker_failure = error

    async def on_composer_submitted(self, message: Composer.Submitted) -> None:
        value = message.value
        user = Text("You › ", style="bold cyan")
        user.append(value)
        await self.transcript.mount(SelectableStatic(user, classes="user-message"))
        self.follow_output(True)
        self.input_reader.submit(value)

    def on_composer_mode_changed(self, _message: Composer.ModeChanged) -> None:
        self.update_context()

    def on_composer_visual_height_changed(
        self,
        _message: Composer.VisualHeightChanged,
    ) -> None:
        self._resize_composer()

    def on_text_area_changed(self, message: TextArea.Changed) -> None:
        if message.text_area is self.composer:
            self._resize_composer()

    def on_resize(self, _event: events.Resize) -> None:
        self.call_after_refresh(self._resize_composer)

    def action_toggle_details(self) -> None:
        self.composer.clear_vim_pending()
        follow = self.transcript.is_vertical_scroll_end
        self.details_expanded = not self.details_expanded
        turns = list(self.query(TurnView))
        current_turn = self.presenter.current_turn
        if current_turn is not None and current_turn not in turns:
            turns.append(current_turn)
        for turn in turns:
            turn.set_details_expanded(self.details_expanded)
        self.follow_output(follow)

    def _copy_selection(self) -> bool:
        selected_text = self.composer.selected_text or self.screen.get_selected_text()
        if selected_text is None:
            return False
        self.copy_to_clipboard(selected_text)
        return True

    def action_copy_selection(self) -> None:
        """Copy app-managed text without giving Command+C a quit fallback."""
        self.composer.clear_vim_pending()
        self._copy_selection()

    def action_cancel_or_quit(self) -> None:
        self.composer.clear_vim_pending()
        if self._copy_selection():
            return
        if not self.cancellation.cancel():
            self.input_reader.submit("/quit")

    def add_widget(self, widget: Static | TurnView) -> None:
        follow = self.transcript.is_vertical_scroll_end
        self.transcript.mount(widget)
        self.follow_output(follow)

    def add_renderable(self, renderable: RenderableType, *, classes: str) -> None:
        self.add_widget(SelectableStatic(renderable, classes=classes))

    def follow_output(self, should_follow: bool) -> None:
        if should_follow:
            self.call_after_refresh(
                self.transcript.scroll_end,
                animate=False,
                immediate=True,
            )

    def update_context(self) -> None:
        estimated_tokens = getattr(self.session, "estimated_context_tokens", 0)
        context_window = getattr(self.session, "context_window_tokens", 0)
        is_estimate = getattr(self.session, "context_tokens_estimated", True)
        compaction_enabled = getattr(self.session, "compaction_enabled", True)
        estimate_marker = "~" if is_estimate else ""
        compaction_marker = "" if compaction_enabled else " · off"
        context_label = (
            f"{estimate_marker}{format_token_count(estimated_tokens)}/"
            f"{format_token_count(context_window)}{compaction_marker}"
            if context_window
            else ""
        )
        mode_label = (
            ("I" if self.composer.vim_mode is VimMode.INSERT else "N")
            if self.vim_mode
            else ""
        )
        label = " · ".join(part for part in (context_label, mode_label) if part)
        self.context.update(Text(label, style="dim"))

    def _resize_composer(self) -> None:
        if not self.composer_shell.is_mounted:
            return
        visual_lines = max(1, self.composer.wrapped_document.height)
        header_height = self.header.region.height or 3
        maximum_height = max(
            COMPOSER_MIN_HEIGHT,
            self.size.height - header_height - TRANSCRIPT_MIN_HEIGHT,
        )
        target_height = min(
            maximum_height,
            max(COMPOSER_MIN_HEIGHT, visual_lines + COMPOSER_BORDER_HEIGHT),
        )
        if target_height != self._composer_height:
            self._composer_height = target_height
            self.composer_shell.styles.height = target_height
            self.call_after_refresh(self._resize_composer)


def _header(app_info: AppInfo) -> Group:
    title = Text("Ghostwheel", style="bold magenta")
    details = Text()
    details.append(f"{app_info.provider}/{app_info.model}", style="cyan")
    details.append("  ·  ")
    details.append(app_info.workspace)
    details.append("  ·  tools: ")
    profile_style = "bold yellow" if app_info.tool_profile == "full" else "green"
    details.append(app_info.tool_profile.upper(), style=profile_style)
    return Group(title, details)


def _help_panel(*, vim_mode: bool = False) -> Panel:
    lines = (
        ("/help", "show commands and keyboard shortcuts"),
        ("/review [path]", "review code; defaults to the workspace"),
        ("/retry", "repeat the previous chat or review"),
        ("/clear", "clear model conversation history"),
        ("/model", "show the active provider and model"),
        ("/tools", "show the active tool profile"),
        ("/quit", "exit Ghostwheel"),
    )
    body = Text()
    for index, (command, description) in enumerate(lines):
        if index:
            body.append("\n")
        body.append(f"{command:<22}", style="bold cyan")
        body.append(description)
    body.append("\n\nShortcuts\n", style="bold")
    body.append("Shift+Enter         insert a newline\n", style="dim")
    body.append("Cmd+C / Ctrl+C     copy selection\n", style="dim")
    body.append("Ctrl+C              otherwise cancel/quit\n", style="dim")
    body.append("Ctrl+O              toggle thinking and tool details\n", style="dim")
    body.append("↑/↓                 recall earlier prompts\n", style="dim")
    body.append("Tab                 complete commands and review paths\n", style="dim")
    body.append("Mouse wheel / bar   scroll the transcript\n", style="dim")
    body.append("Mouse drag          select transcript text\n", style="dim")
    body.append("Right-click         copy selected text", style="dim")
    if vim_mode:
        body.append("\n\nVim prompt editing\n", style="bold")
        body.append("Esc / i a I A       switch Normal / Insert mode\n", style="dim")
        body.append("h j k l · w b e · 0 $   move the cursor\n", style="dim")
        body.append("x X · d c y + motion    edit or copy text\n", style="dim")
        body.append("o O · p P · u Ctrl+R    open, paste, undo/redo", style="dim")
    return Panel(body, title="Commands", border_style="cyan")


def _tool_summary(activity: ToolActivity) -> Text:
    icon, style = {
        "running": ("▸", "yellow"),
        "succeeded": ("✓", "green"),
        "failed": ("✗", "red"),
    }[activity.status]
    line = Text(f"  {icon} ", style=style)
    line.append(activity.name, style=f"bold {style}")
    argument = primary_argument(activity.arguments)
    if argument:
        line.append("  ")
        line.append(preview(argument, 72))
    if activity.finished_at is not None:
        line.append("  ·  ", style="dim")
        line.append(duration(activity.finished_at - activity.started_at), style="dim")
    if activity.status == "failed" and activity.detail:
        line.append("  ·  ", style="red")
        line.append(
            preview(" ".join(activity.detail.split()), 100),
            style="red",
        )
    return line


def _tool_detail(activity: ToolActivity) -> Panel:
    body = Text()
    body.append("Arguments\n", style="bold")
    body.append(activity.arguments or "(none)")
    if activity.detail:
        body.append("\n\nResult\n", style="bold")
        body.append(activity.detail)
    return Panel(
        body,
        title=Text(f"{activity.name} details", style="dim"),
        border_style="dim",
        padding=(0, 1),
    )


def _turn_failure(outcome: TurnFailed) -> Panel:
    presentation = failure_presentation(outcome.kind)
    body = Text(outcome.message)
    body.append(f"\n\n{presentation.hint}", style="dim")
    return Panel(body, title=presentation.title, border_style="red")
