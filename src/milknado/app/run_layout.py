"""Responsive layout constants for shared run and watch views."""

WIDE_MIN_COLUMNS = 116
MINIMUM_FALLBACK_TEXT = (
    "Terminal too small for session controls. Resize to at least 60x18.\nq quit · ? help"
)
RUN_VIEW_CSS = """
Screen { layers: base overlay; }
#workspace { height: 1fr; layer: base; }
#events {
    height: auto; min-height: 4; max-height: 5;
    margin: 0 1; border: round $secondary; layer: base;
}
#minimum-fallback {
    display: none;
    dock: top;
    layer: overlay;
    width: 1fr;
    height: auto;
    margin: 1 1;
    padding: 0 1;
    border: round $warning;
    content-align: center middle;
    text-align: center;
}
Header, Footer { layer: base; }
#open-hint {
    display: none;
    dock: right;
    width: auto;
    height: 1;
    padding: 0 1;
    background: $footer-background;
}
.compact.list #open-hint { display: block; }
#detail #help { display: none; }
.compact #workspace { display: block; }
.compact #totals { display: block; }
.compact #run-panel { width: 1fr; }
.compact.list #detail { display: none; }
.compact.detail #run-panel { display: none; }
.compact.detail #events, .compact #session-state { display: none; }
.minimum Header, .minimum Footer, .minimum #workspace, .minimum #events {
    display: none;
}
.minimum #minimum-fallback { display: block; }
"""
