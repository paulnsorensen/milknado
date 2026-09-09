Set Shell "bash"
Set Width 520
Set Height 420
Set FontSize 16
Set Theme "Catppuccin Mocha"
Set TypingSpeed 0ms
Hide
Type "unset PROMPT_COMMAND; PS1='VHS_READY>'"
Enter
Wait+Screen@5s /VHS_READY>/
Type "env -u NO_COLOR COLORTERM=truecolor uv run python docs/tui-captures/parity-hitl/fixture.py run error"
Enter
Wait+Screen@20s /Milknado/

Show
Sleep 300ms
Screenshot docs/tui-captures/parity-hitl/run-error-520x420-after.png
Sleep 500ms
Hide
Ctrl+c
Wait+Screen@10s /confirm/
Type "y"
Wait+Screen@10s /VHS_READY>/
Type "exit"
Enter
