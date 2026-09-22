"""Shared footer and help overlays for run and watch workspaces."""

from __future__ import annotations

from typing import ClassVar, Protocol, TypeAlias, cast, final

from textual.app import App, ComposeResult
from textual.binding import ActiveBinding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.dom import DOMNode
from textual.events import Click, Resize
from textual.geometry import Size
from textual.screen import ModalScreen
from textual.widgets import Footer, Static
from typing_extensions import override

_Hint: TypeAlias = tuple[str, str, DOMNode]


class _FooterHost(Protocol):
    active_bindings: dict[str, ActiveBinding]
    size: Size

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None: ...

    def get_key_display(self, binding: object) -> str: ...


@final
class FooterHint(Static):
    """Clickable compact footer action."""

    def __init__(self, label: str, action: str, owner: DOMNode) -> None:
        super().__init__(label, markup=False)
        self._action: str = action
        self._owner: DOMNode = owner

    async def on_click(self, event: Click) -> None:
        _ = event.stop()
        host = cast(App[object], cast(object, self.app))
        _ = await host.run_action(self._action, default_namespace=self._owner)


class RunFooter(Footer):
    """Keep contextual controls readable in the mounted footer."""

    DEFAULT_CSS: ClassVar[str] = """
    RunFooter { height: auto; min-height: 1; }
    #footer-hints { width: 1fr; height: auto; min-height: 1; }
    .footer-row { width: 100%; height: 1; align: center middle; }
    FooterHint { width: auto; height: 1; padding: 0 1; }
    FooterHint:hover { background: $boost; }
    """
    _hint_width: int | None = None
    _hint_signature: tuple[_Hint, ...] = ()

    @override
    def compose(self) -> ComposeResult:
        hints = self._available_hints()
        host = cast(_FooterHost, cast(object, self.app))
        width = max(host.size.width - 2, 1)
        self._hint_width = width
        self._hint_signature = hints
        with Vertical(id="footer-hints"):
            for row in self._pack_hints(hints, width):
                with Horizontal(classes="footer-row"):
                    for label, action, owner in row:
                        yield FooterHint(label, action, owner)

    def _available_hints(self) -> tuple[_Hint, ...]:
        host = cast(_FooterHost, cast(object, self.app))
        hints: list[_Hint] = []
        seen: set[tuple[DOMNode, str]] = set()
        for active in host.active_bindings.values():
            binding = active.binding
            owner_action = (active.node, binding.action)
            if not binding.show or owner_action in seen or not active.enabled:
                continue
            key = host.get_key_display(binding)
            if key in {"enter", "⏎", "esc"}:
                key = "Enter" if key in {"enter", "⏎"} else "Esc"
            if not binding.description:
                continue
            hints.append((f"{key} {binding.description}", binding.action, active.node))
            seen.add(owner_action)
        owner = cast(DOMNode, cast(object, host))
        if (owner, "open_detail") not in seen and host.check_action("open_detail", ()) is True:
            enter = next(
                (
                    host.get_key_display(active.binding)
                    for active in host.active_bindings.values()
                    if active.binding.key == "enter"
                ),
                "Enter",
            )
            enter_label = "Enter" if enter in {"enter", "⏎"} else enter.capitalize()
            hints.append((f"{enter_label} Open", "open_detail", owner))
        return tuple(hints)

    @staticmethod
    def _pack_hints(hints: tuple[_Hint, ...], width: int) -> tuple[tuple[_Hint, ...], ...]:
        rows: list[list[_Hint]] = []
        row: list[_Hint] = []
        row_width = 0
        for hint in hints:
            needed = len(hint[0]) + (3 if row else 0)
            if row and row_width + needed > width:
                rows.append(row)
                row = []
                row_width = 0
                needed = len(hint[0])
            row.append(hint)
            row_width += needed
        if row:
            rows.append(row)
        return tuple(tuple(row) for row in rows)

    def on_resize(self, event: Resize) -> None:
        self.update_hints(max(event.size.width - 2, 1))

    def update_hints(self, width: int | None = None) -> None:
        host = cast(_FooterHost, cast(object, self.app))
        width = max(host.size.width - 2, 1) if width is None else width
        hints = self._available_hints()
        if width != self._hint_width or hints != self._hint_signature:
            self._hint_width = width
            self._hint_signature = hints
            _ = self.call_after_refresh(self.recompose)


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
