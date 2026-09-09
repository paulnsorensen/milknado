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
Type "env -u NO_COLOR COLORTERM=truecolor uv run python docs/tui-captures/parity-hitl/fixture.py watch error"
Enter
Wait+Screen@20s /Milknado/

Show
Sleep 300ms
Screenshot docs/tui-captures/parity-hitl/watch-error-520x420-before.png
Sleep 500ms
Hide
Type "q"
Wait+Screen@10s /VHS_READY>/
Type "exit"
Enter
