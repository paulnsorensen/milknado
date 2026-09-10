Output .cheese/tui-demo/structured-sessions/before.gif
Set Shell "bash"
Set Width 1200
Set Height 720
Set FontSize 16
Set Theme "Catppuccin Mocha"
Set TypingSpeed 0ms
Hide
Type "cd /tmp/milknado-structured-demo && env -u NO_COLOR TERM=xterm-256color COLORTERM=truecolor PYTHONPATH=/tmp/milknado-structured-baseline/src uv run python docs/tui-captures/structured-sessions/before.py && printf '\nMILKNADO_UI_EXITED\n'"
Enter
Wait+Screen@30s /Node 12 started/
Enter
Wait+Screen@15s /"type": "message_end"/
Show
Sleep 750ms
Screenshot .cheese/tui-demo/structured-sessions/before.png
Type "q"
Wait+Screen@15s /Stop scheduling/
Type "y"
Wait+Screen@15s /0 active/
Type "q"
Wait+Screen@15s /MILKNADO_UI_EXITED/
Hide
Type "exit"
Enter
