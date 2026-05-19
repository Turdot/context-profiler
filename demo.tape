Output assets/readme/demo.gif

Set Shell zsh
Set FontSize 14
Set Width 960
Set Height 540
Set Theme "Catppuccin Mocha"
Set Padding 20

Type "context-profiler analyze examples/swe_agent/session.jsonl --format openai"
Sleep 500ms
Enter
Sleep 3s

Sleep 2s

Type "context-profiler diagnose examples/swe_agent/session.jsonl --format openai --json | jq '.issues[:2]'"
Sleep 500ms
Enter
Sleep 3s

Sleep 2s
