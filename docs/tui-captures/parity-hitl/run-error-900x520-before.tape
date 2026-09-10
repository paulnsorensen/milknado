Set Shell "bash"
Set Width 900
Set Height 520
Set FontSize 16
Set Theme "Catppuccin Mocha"
Set TypingSpeed 0ms
Hide
Type "unset PROMPT_COMMAND; PS1='VHS_READY>'"
Enter
Wait+Screen@5s /VHS_READY>/
Type "env -u NO_COLOR COLORTERM=truecolor uv run python docs/tui-captures/parity-hitl/fixture.py run error"
Enter
Wait+Screen@20s /Repair compact/

Show
Sleep 300ms
Screenshot docs/tui-captures/parity-hitl/run-error-900x520-before.png
Sleep 500ms
Hide
Ctrl+c
Wait+Screen@10s /gracefully stop/
Type "y"
Wait+Screen@10s /VHS_READY>/
Type "exit"
Enter
