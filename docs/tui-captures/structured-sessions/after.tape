Output .cheese/tui-demo/structured-sessions/after.gif
Set Shell "bash"
Set Width 1200
Set Height 720
Set FontSize 16
Set Theme "Catppuccin Mocha"
Set TypingSpeed 0ms
Hide
Type "cd /tmp/milknado-structured-demo && env -u NO_COLOR TERM=xterm-256color COLORTERM=truecolor uv run python docs/tui-captures/structured-sessions/fixture.py && printf '\nMILKNADO_UI_EXITED\n'"
Enter
Wait+Screen@30s /Node 12 started/
Enter
Wait+Screen@15s /Watch stays read-only/
Show
Sleep 750ms
Screenshot .cheese/tui-demo/structured-sessions/after.png
Type "i"
Wait+Screen@15s /esc Back.*f1 Help/
Type "receipt-proof-2026"
Wait+Screen@15s /receipt-proof-2026/
Enter
Wait+Screen@15s /\[delivered\]: receipt-proof-2026/
Sleep 750ms
Screenshot .cheese/tui-demo/structured-sessions/after-input.png
Escape
Wait+Screen@15s /\? Help/
Type "x"
Wait+Screen@15s /2 changed files/
Wait+Screen@15s /Preserve permission policy and completion gates/
Sleep 750ms
Screenshot .cheese/tui-demo/structured-sessions/after-changes.png
Shift+Tab
Right
Wait+Screen@15s /Decode OMP/
Sleep 750ms
Screenshot .cheese/tui-demo/structured-sessions/after-details.png
Type "?"
Wait+Screen@15s /Esc\/F1 close/
Sleep 750ms
Screenshot .cheese/tui-demo/structured-sessions/after-help.png
Escape
Wait+Screen@15s /Decode OMP/
Ctrl+p
Type "quit"
Wait+Screen@15s /Quit the application/
Sleep 750ms
Screenshot .cheese/tui-demo/structured-sessions/after-search.png
Enter
Wait+Screen@15s /Stop scheduling/
Sleep 750ms
Screenshot .cheese/tui-demo/structured-sessions/after-confirm.png
Type "y"
Wait+Screen@15s /0 active/
Type "q"
Wait+Screen@15s /MILKNADO_UI_EXITED/
Hide
Type "exit"
Enter
