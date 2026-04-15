"""Minimal interactive prompts for the SimLab CLI.

This is a small, self-contained, questionary-style prompt layer implemented with
only the Python standard library. It is intentionally narrow and currently used
to drive the autoresearch UX without adding new third-party dependencies.
"""

from __future__ import annotations

import contextlib
import shutil
import string
import sys
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic
from typing import TextIO
from typing import TypeVar

try:
    import termios
    import tty
except ImportError:  # pragma: no cover
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

T = TypeVar("T")


@dataclass(frozen=True)
class Choice(Generic[T]):
    """One choice rendered in a select/checkbox prompt."""

    title: str
    value: T


@dataclass(frozen=True)
class PromptTheme:
    """Display strings for prompt widgets."""

    pointer: str = ">"
    checked: str = "[x]"
    unchecked: str = "[ ]"


class Prompt(Generic[T]):
    """Base prompt type with a questionary-style `ask()` entrypoint."""

    def ask(
        self,
        *,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> T:
        """Prompt the user and return the answer."""
        raise NotImplementedError


@dataclass(frozen=True)
class TextPrompt(Prompt[str]):
    """Read one line of text with optional default and validation."""

    message: str
    default: str | None = None
    validate: Callable[[str], str | None] | None = None

    def ask(self, *, stdin: TextIO | None = None, stdout: TextIO | None = None) -> str:
        """Return a validated string answer."""
        input_stream = stdin or sys.stdin
        output_stream = stdout or sys.stdout
        needs_leading_gap = True

        while True:
            if needs_leading_gap:
                output_stream.write("\n")
                output_stream.flush()
                needs_leading_gap = False
            if getattr(output_stream, "isatty", lambda: False)():
                # Ensure we start at column 0 on a clean line, even if the terminal
                # treated the prior Enter key as newline without carriage return.
                output_stream.write("\x1b[2K\r")
            suffix = "" if self.default is None else f" [{self.default}]"
            output_stream.write(f"{self.message}{suffix}: ")
            output_stream.flush()
            try:
                raw = input_stream.readline()
            except KeyboardInterrupt as exc:
                raise KeyboardInterrupt from exc
            if raw == "":
                raise EOFError("stdin closed")
            if getattr(output_stream, "isatty", lambda: False)():
                # Some terminals treat the Enter key as newline without carriage return,
                # which can cause later output to start at an indented column.
                output_stream.write("\r")
                output_stream.flush()
            answer = raw.rstrip("\n")
            if not answer and self.default is not None:
                answer = self.default

            if self.validate is None:
                return answer
            error = self.validate(answer)
            if error is None:
                return answer

            output_stream.write(f"{error}\n")
            output_stream.flush()


@dataclass(frozen=True)
class ConfirmPrompt(Prompt[bool]):
    """Read a yes/no answer with a default."""

    message: str
    default: bool = False

    def ask(self, *, stdin: TextIO | None = None, stdout: TextIO | None = None) -> bool:
        """Return True/False based on a y/n answer."""
        input_stream = stdin or sys.stdin
        output_stream = stdout or sys.stdout

        default_hint = "Y/n" if self.default else "y/N"
        needs_leading_gap = True
        while True:
            if needs_leading_gap:
                output_stream.write("\n")
                output_stream.flush()
                needs_leading_gap = False
            if getattr(output_stream, "isatty", lambda: False)():
                output_stream.write("\x1b[2K\r")
            output_stream.write(f"{self.message} [{default_hint}]: ")
            output_stream.flush()
            raw = input_stream.readline()
            if raw == "":
                raise EOFError("stdin closed")
            if getattr(output_stream, "isatty", lambda: False)():
                output_stream.write("\r")
                output_stream.flush()
            answer = raw.strip().lower()
            if not answer:
                return self.default
            if answer in {"y", "yes"}:
                return True
            if answer in {"n", "no"}:
                return False
            output_stream.write("Please enter y or n.\n")
            output_stream.flush()


@dataclass(frozen=True)
class SelectPrompt(Prompt[T]):
    """Interactive single-choice selection prompt."""

    message: str
    choices: Sequence[Choice[T]]
    default: T | None = None
    theme: PromptTheme = PromptTheme()
    instruction: str = "Use Up/Down to navigate, type to filter, Enter to select."

    def ask(self, *, stdin: TextIO | None = None, stdout: TextIO | None = None) -> T:
        """Return one selected choice value."""
        input_stream = stdin or sys.stdin
        output_stream = stdout or sys.stdout

        if not is_tty(input_stream, output_stream):
            return select_fallback(
                message=self.message,
                choices=self.choices,
                default=self.default,
                stdin=input_stream,
                stdout=output_stream,
            )

        return run_select(
            message=self.message,
            choices=self.choices,
            default=self.default,
            theme=self.theme,
            instruction=self.instruction,
            stdin=input_stream,
            stdout=output_stream,
        )


@dataclass(frozen=True)
class CheckboxPrompt(Prompt[list[T]]):
    """Interactive multi-choice selection prompt."""

    message: str
    choices: Sequence[Choice[T]]
    default: Sequence[T] = ()
    min_selected: int = 0
    select_all_value: T | None = None
    theme: PromptTheme = PromptTheme()
    instruction: str = (
        "Use Up/Down to navigate, Space to toggle, type to filter, Enter to continue."
    )

    def ask(self, *, stdin: TextIO | None = None, stdout: TextIO | None = None) -> list[T]:
        """Return selected values in their original order."""
        input_stream = stdin or sys.stdin
        output_stream = stdout or sys.stdout

        if not is_tty(input_stream, output_stream):
            return checkbox_fallback(
                message=self.message,
                choices=self.choices,
                default=self.default,
                min_selected=self.min_selected,
                select_all_value=self.select_all_value,
                stdin=input_stream,
                stdout=output_stream,
            )

        return run_checkbox(
            message=self.message,
            choices=self.choices,
            default=self.default,
            min_selected=self.min_selected,
            select_all_value=self.select_all_value,
            theme=self.theme,
            instruction=self.instruction,
            stdin=input_stream,
            stdout=output_stream,
        )


def text(
    message: str,
    *,
    default: str | None = None,
    validate: Callable[[str], str | None] | None = None,
) -> TextPrompt:
    """Create a text prompt."""
    return TextPrompt(message=message, default=default, validate=validate)


def confirm(message: str, *, default: bool = False) -> ConfirmPrompt:
    """Create a yes/no prompt."""
    return ConfirmPrompt(message=message, default=default)


def select(
    message: str,
    *,
    choices: Sequence[Choice[T]],
    default: T | None = None,
    instruction: str | None = None,
) -> SelectPrompt[T]:
    """Create a single-choice prompt."""
    prompt = SelectPrompt(message=message, choices=choices, default=default)
    if instruction is None:
        return prompt
    return SelectPrompt(
        message=prompt.message,
        choices=prompt.choices,
        default=prompt.default,
        theme=prompt.theme,
        instruction=instruction,
    )


def checkbox(
    message: str,
    *,
    choices: Sequence[Choice[T]],
    default: Sequence[T] = (),
    min_selected: int = 0,
    select_all_value: T | None = None,
    instruction: str | None = None,
) -> CheckboxPrompt[T]:
    """Create a multi-choice prompt."""
    prompt = CheckboxPrompt(
        message=message,
        choices=choices,
        default=default,
        min_selected=min_selected,
        select_all_value=select_all_value,
    )
    if instruction is None:
        return prompt
    return CheckboxPrompt(
        message=prompt.message,
        choices=prompt.choices,
        default=prompt.default,
        min_selected=prompt.min_selected,
        select_all_value=prompt.select_all_value,
        theme=prompt.theme,
        instruction=instruction,
    )


def is_tty(stdin: TextIO, stdout: TextIO) -> bool:
    """Return whether we can run an interactive prompt on these streams."""
    if termios is None or tty is None:
        return False
    return bool(getattr(stdin, "isatty", lambda: False)()) and bool(
        getattr(stdout, "isatty", lambda: False)()
    )


def select_fallback(
    *,
    message: str,
    choices: Sequence[Choice[T]],
    default: T | None,
    stdin: TextIO,
    stdout: TextIO,
) -> T:
    """Fallback selection prompt for non-TTY usage."""
    if not choices:
        raise ValueError("select requires at least one choice")

    labels = "\n".join(f"  {idx}. {choice.title}" for idx, choice in enumerate(choices, start=1))
    stdout.write(f"\n{message}\n{labels}\n")
    stdout.flush()

    default_index: int | None = None
    if default is not None:
        for idx, choice in enumerate(choices, start=1):
            if choice.value == default:
                default_index = idx
                break

    prompt_default = "" if default_index is None else str(default_index)
    raw = TextPrompt(message="Choose (number)", default=prompt_default).ask(
        stdin=stdin,
        stdout=stdout,
    )
    raw = raw.strip()
    if not raw.isdigit():
        raise ValueError("expected numeric choice")
    picked = int(raw)
    if picked < 1 or picked > len(choices):
        raise ValueError("choice out of range")
    return choices[picked - 1].value


def checkbox_fallback(
    *,
    message: str,
    choices: Sequence[Choice[T]],
    default: Sequence[T],
    min_selected: int,
    select_all_value: T | None,
    stdin: TextIO,
    stdout: TextIO,
) -> list[T]:
    """Fallback checkbox prompt for non-TTY usage."""
    if not choices:
        raise ValueError("checkbox requires at least one choice")

    labels = "\n".join(f"  {idx}. {choice.title}" for idx, choice in enumerate(choices, start=1))
    stdout.write(f"\n{message}\n{labels}\n")
    stdout.flush()

    default_hint = ""
    if default:
        default_indexes: list[str] = []
        for idx, choice in enumerate(choices, start=1):
            if choice.value in set(default):
                default_indexes.append(str(idx))
        default_hint = ",".join(default_indexes)

    raw = TextPrompt(message="Choose (numbers, comma-separated)", default=default_hint).ask(
        stdin=stdin, stdout=stdout
    )
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    picked_indexes: list[int] = []
    for token in tokens:
        if not token.isdigit():
            continue
        idx = int(token)
        if 1 <= idx <= len(choices):
            picked_indexes.append(idx)

    picked_values = [choices[idx - 1].value for idx in picked_indexes]
    if select_all_value is not None and select_all_value in picked_values:
        picked_values = [c.value for c in choices if c.value != select_all_value]
    if len(picked_values) < min_selected:
        raise ValueError(f"pick at least {min_selected} item(s)")
    return picked_values


@contextlib.contextmanager
def raw_tty(stdin: TextIO, stdout: TextIO) -> Iterator[Callable[[str], None]]:
    """Put stdin into raw mode and yield a write function for stdout."""
    if termios is None or tty is None:
        raise RuntimeError("raw terminal mode is not available on this platform")

    fd = stdin.fileno()
    old = termios.tcgetattr(fd)

    def write(text: str) -> None:
        stdout.write(text)
        stdout.flush()

    try:
        tty.setraw(fd)
        write("\x1b[?25l")
        yield write
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        write("\x1b[?25h")


def read_key(stdin: TextIO) -> str:
    """Read one keypress from a raw TTY and return a normalized key name."""
    ch = stdin.read(1)
    if ch == "":
        return "eof"
    if ch == "\x03":
        return "ctrl_c"
    if ch in {"\r", "\n"}:
        return "enter"
    if ch == "\x7f":
        return "backspace"
    if ch == " ":
        return "space"
    if ch == "\x1b":
        nxt = stdin.read(1)
        if nxt == "[":
            code = stdin.read(1)
            if code == "A":
                return "up"
            if code == "B":
                return "down"
        return "escape"
    if ch in {"j", "J"}:
        return "down"
    if ch in {"k", "K"}:
        return "up"
    if ch in string.printable and ch not in {"\x0b", "\x0c"}:
        return f"char:{ch}"
    return "unknown"


def render_lines(
    *,
    message: str,
    instruction: str,
    choices: Sequence[Choice[T]],
    selected_index: int,
    checkbox_selected: set[T] | None,
    theme: PromptTheme,
    query: str,
    scroll_start: int,
    page_size: int,
) -> list[str]:
    """Return screen lines representing the current prompt state."""
    width = shutil.get_terminal_size(fallback=(80, 24)).columns

    lines: list[str] = []
    lines.append("")
    lines.append(f"{message}")
    lines.append(instruction)
    if query:
        lines.append(f"Filter: {query}")
    else:
        lines.append("Filter: (type to search)")

    visible = list(choices)[scroll_start : scroll_start + page_size]
    if not visible:
        lines.append("  (no matches)")
        return lines

    for idx, choice in enumerate(visible, start=scroll_start):
        pointer = theme.pointer if idx == selected_index else " "
        mark = ""
        if checkbox_selected is not None:
            mark = theme.checked if choice.value in checkbox_selected else theme.unchecked
            mark = f"{mark} "

        title = choice.title.replace("\n", " ").strip()
        prefix = f" {pointer} {mark}"
        remaining = max(0, width - len(prefix))
        if remaining and len(title) > remaining:
            title = title[: max(0, remaining - 3)] + "..."

        line = f"{prefix}{title}"
        if idx == selected_index:
            line = f"\x1b[7m{line}\x1b[0m"
        lines.append(line)

    return lines


def move_cursor_up(write: Callable[[str], None], lines: int) -> None:
    """Move the cursor up by N lines."""
    if lines <= 0:
        return
    write(f"\x1b[{lines}A")


def clear_line(write: Callable[[str], None]) -> None:
    """Clear the current line."""
    write("\x1b[2K\r")


def render_screen(
    write: Callable[[str], None],
    *,
    lines: list[str],
    previous_line_count: int,
) -> int:
    """Re-render the prompt screen and return the number of printed lines."""
    move_cursor_up(write, previous_line_count)
    for line in lines:
        clear_line(write)
        write(line + "\r\n")
    write("\x1b[0J")
    return len(lines)


def run_select(
    *,
    message: str,
    choices: Sequence[Choice[T]],
    default: T | None,
    theme: PromptTheme,
    instruction: str,
    stdin: TextIO,
    stdout: TextIO,
) -> T:
    """Run an interactive select prompt."""
    if not choices:
        raise ValueError("select requires at least one choice")

    stdout.write("\n")
    stdout.flush()

    selected_index = 0
    if default is not None:
        for idx, choice in enumerate(choices):
            if choice.value == default:
                selected_index = idx
                break

    query = ""
    scroll_start = 0
    line_count = 0

    with raw_tty(stdin, stdout) as write:
        while True:
            visible = [c for c in choices if query_match(query=query, candidate=c.title)]

            size = shutil.get_terminal_size(fallback=(80, 24))
            page_size = max(4, size.lines - 8)
            selected_index = min(selected_index, len(visible) - 1) if visible else 0

            scroll_start = clamp_scroll_start(
                selected_index=selected_index,
                scroll_start=scroll_start,
                page_size=page_size,
            )

            screen = render_lines(
                message=message,
                instruction=instruction,
                choices=visible,
                selected_index=selected_index,
                checkbox_selected=None,
                theme=theme,
                query=query,
                scroll_start=scroll_start,
                page_size=page_size,
            )
            line_count = render_screen(write, lines=screen, previous_line_count=line_count)

            key = read_key(stdin)
            if key == "ctrl_c":
                raise KeyboardInterrupt
            if key == "escape":
                raise KeyboardInterrupt
            if key == "up":
                selected_index = max(0, selected_index - 1)
                continue
            if key == "down":
                selected_index = min(max(0, len(visible) - 1), selected_index + 1)
                continue
            if key == "backspace":
                query = query[:-1]
                selected_index = 0
                scroll_start = 0
                continue
            if key.startswith("char:"):
                ch = key.split(":", 1)[1]
                if ch in string.whitespace:
                    continue
                query += ch
                selected_index = 0
                scroll_start = 0
                continue
            if key == "enter":
                if not visible:
                    continue
                picked = visible[selected_index]
                final = f"{message}: {picked.title}\n"
                line_count = render_screen(
                    write,
                    lines=[final.rstrip("\n")],
                    previous_line_count=line_count,
                )
                return picked.value


def run_checkbox(
    *,
    message: str,
    choices: Sequence[Choice[T]],
    default: Sequence[T],
    min_selected: int,
    select_all_value: T | None,
    theme: PromptTheme,
    instruction: str,
    stdin: TextIO,
    stdout: TextIO,
) -> list[T]:
    """Run an interactive checkbox prompt."""
    if not choices:
        raise ValueError("checkbox requires at least one choice")

    stdout.write("\n")
    stdout.flush()

    selected: set[T] = set(default)
    selected_index = 0
    query = ""
    scroll_start = 0
    line_count = 0
    select_all_choice = (
        next((c for c in choices if c.value == select_all_value), None)
        if select_all_value is not None
        else None
    )
    non_select_all_choices = (
        [c for c in choices if c.value != select_all_value]
        if select_all_value is not None
        else list(choices)
    )
    all_values = [c.value for c in non_select_all_choices]

    with raw_tty(stdin, stdout) as write:
        while True:
            if all_values and all(v in selected for v in all_values):
                if select_all_value is not None:
                    selected.add(select_all_value)
            elif select_all_value is not None:
                selected.discard(select_all_value)

            visible: list[Choice[T]] = []
            if select_all_choice is not None:
                visible.append(select_all_choice)
            visible.extend(
                [c for c in non_select_all_choices if query_match(query=query, candidate=c.title)]
            )

            size = shutil.get_terminal_size(fallback=(80, 24))
            page_size = max(4, size.lines - 9)
            selected_index = min(selected_index, len(visible) - 1) if visible else 0

            scroll_start = clamp_scroll_start(
                selected_index=selected_index,
                scroll_start=scroll_start,
                page_size=page_size,
            )

            selected_count = (
                len([v for v in selected if select_all_value is None or v != select_all_value])
                if selected
                else 0
            )
            header = f"{instruction}  ({selected_count} selected)"
            screen = render_lines(
                message=message,
                instruction=header,
                choices=visible,
                selected_index=selected_index,
                checkbox_selected=set(selected),
                theme=theme,
                query=query,
                scroll_start=scroll_start,
                page_size=page_size,
            )
            line_count = render_screen(write, lines=screen, previous_line_count=line_count)

            key = read_key(stdin)
            if key == "ctrl_c":
                raise KeyboardInterrupt
            if key == "escape":
                raise KeyboardInterrupt
            if key == "up":
                selected_index = max(0, selected_index - 1)
                continue
            if key == "down":
                selected_index = min(max(0, len(visible) - 1), selected_index + 1)
                continue
            if key == "space":
                if not visible:
                    continue
                picked = visible[selected_index]
                if select_all_value is not None and picked.value == select_all_value:
                    if all_values and all(v in selected for v in all_values):
                        for v in all_values:
                            selected.discard(v)
                        selected.discard(select_all_value)
                    else:
                        for v in all_values:
                            selected.add(v)
                        selected.add(select_all_value)
                    continue
                if picked.value in selected:
                    selected.remove(picked.value)
                else:
                    selected.add(picked.value)
                continue
            if key == "backspace":
                query = query[:-1]
                selected_index = 0
                scroll_start = 0
                continue
            if key.startswith("char:"):
                ch = key.split(":", 1)[1]
                if ch in string.whitespace:
                    continue
                query += ch
                selected_index = 0
                scroll_start = 0
                continue
            if key == "enter":
                if select_all_value is not None:
                    picked_values = [
                        c.value
                        for c in choices
                        if c.value in selected and c.value != select_all_value
                    ]
                else:
                    picked_values = [c.value for c in choices if c.value in selected]
                if len(picked_values) < min_selected:
                    continue
                final = f"{message}: {', '.join(str(v) for v in picked_values)}\n"
                line_count = render_screen(
                    write,
                    lines=[final.rstrip("\n")],
                    previous_line_count=line_count,
                )
                return picked_values


def query_match(*, query: str, candidate: str) -> bool:
    """Return whether candidate matches the current filter query."""
    q = query.strip().lower()
    if not q:
        return True
    return q in candidate.lower()


def clamp_scroll_start(*, selected_index: int, scroll_start: int, page_size: int) -> int:
    """Clamp scroll window so the selection stays visible."""
    if page_size <= 0:
        return 0
    if selected_index < scroll_start:
        return selected_index
    if selected_index >= scroll_start + page_size:
        return max(0, selected_index - page_size + 1)
    return max(0, scroll_start)
