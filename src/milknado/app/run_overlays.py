"""Shared footer and help overlays for run and watch workspaces."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import BindingType
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Static
from typing_extensions import override


class RunFooter(Footer):
    """Keep the compact open action visible despite DataTable's Enter binding."""

    @override
    def compose(self) -> ComposeResult:
        yield from super().compose()
        yield Static("Enter Open", id="open-hint", markup=False)


class HelpScreen(ModalScreen[None]):
    """Scrollable key guidance for the shared workspace."""

    SCOPED_CSS: ClassVar[bool] = False  # noqa: V107 - Textual class configuration
    AUTO_FOCUS: ClassVar[str | None] = "#help-scroll"  # noqa: V107 - Textual initial focus
    BINDINGS: ClassVar[list[BindingType]] = [  # noqa: V107 - Textual key bindings
        (key, "close_help", "Close") for key in ("escape", "f1", "?", "h", "q")
    ]
    DEFAULT_CSS: ClassVar[str] = """
    #session-help { align: center middle; background: $background 75%; }
    #help-scroll {
        width: 90%; max-width: 96; height: 90%; max-height: 32;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #help-close { dock: bottom; height: 1; text-align: center; background: $surface; }
    """

    def __init__(self, body: str) -> None:
        super().__init__(id="session-help")
        self._body: str = body

    @override
    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help-scroll"):
            yield Static(self._body, id="help-overlay", markup=False)
        yield Static("↑/↓ scroll · Esc/F1 close", id="help-close", markup=False)

    def action_close_help(self) -> None:  # noqa: V105 - Textual binding action
        _ = self.dismiss(None)
