' Desktop shortcut target: opens TTS Studio as a desktop window (no console).
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
py = ""
For Each cand In Array(root & "\.venv\Scripts\pythonw.exe", root & "\..\breeze-tts\.venv\Scripts\pythonw.exe")
  If py = "" And fso.FileExists(cand) Then py = cand
Next
If py = "" Then
  MsgBox "Python not found. Run install.bat first.", 48, "TTS Studio"
  WScript.Quit 1
End If
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = root
sh.Run """" & py & """ """ & root & "\desktop.py""", 1, False
