Set Shell "bash"
Set Width 1400
Set Height 850
Set FontSize 16
Set Theme "Catppuccin Mocha"
Set TypingSpeed 0ms
Hide
Type "unset PROMPT_COMMAND; PS1='VHS_READY>'"
Enter
Wait+Screen@5s /VHS_READY>/
Type "env -u NO_COLOR COLORTERM=truecolor uv run python docs/tui-captures/parity-hitl/fixture.py run escape"
Enter
Wait+Screen@20s /Repair compact/
Type "g"
Wait+Screen@10s /Queue guidance/
Sleep 300ms
Type "Keep the quality gate"
Escape
Sleep 300ms
Show
Sleep 300ms
Screenshot docs/tui-captures/parity-hitl/run-escape-1400x850-after.png
Sleep 500ms
Hide
Ctrl+c
Wait+Screen@10s /confirm/
Type "y"
Wait+Screen@10s /VHS_READY>/
Type "exit"
Enter
