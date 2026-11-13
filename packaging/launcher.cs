// TTS Studio.exe：安装目录里的启动器。双击 = 用自带的 Python 运行 boot.py；带 --uninstall = 卸载。
// 用 Windows 自带的 csc.exe（.NET Framework 4）编译，只能用 C# 5 语法。
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Windows.Forms;
using Microsoft.Win32;

[assembly: System.Reflection.AssemblyTitle("TTS Studio")]
[assembly: System.Reflection.AssemblyProduct("TTS Studio")]

static class Launcher
{
    const string Name = "TTS Studio";

    [STAThread]
    static int Main(string[] args)
    {
        string root = Path.GetDirectoryName(Application.ExecutablePath);
        if (args.Length > 0 && args[0] == "--uninstall")
            return Uninstall(root);

        string py = Path.Combine(root, @"runtime\python\pythonw.exe");
        string boot = Path.Combine(root, "boot.py");
        if (!File.Exists(py) || !File.Exists(boot))
        {
            MessageBox.Show("程序文件不完整，请重新运行安装包。", Name, MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
        var psi = new ProcessStartInfo(py, "\"" + boot + "\"");
        psi.WorkingDirectory = root;
        psi.UseShellExecute = false;
        Process.Start(psi);
        return 0;
    }

    static void KillOwned(string root)
    {
        foreach (var p in Process.GetProcesses())
        {
            try
            {
                string f = p.MainModule.FileName;
                if (f.StartsWith(root, StringComparison.OrdinalIgnoreCase) && p.Id != Process.GetCurrentProcess().Id)
                    p.Kill();
            }
            catch { }
        }
    }

    static int Uninstall(string root)
    {
        if (MessageBox.Show("确定要卸载 TTS Studio 吗？", Name, MessageBoxButtons.YesNo, MessageBoxIcon.Question) != DialogResult.Yes)
            return 0;
        var keep = MessageBox.Show("要保留你的音色、项目、生成的音频和已下载的模型吗？\n\n" +
                                   "选「是」：保留在 " + Path.Combine(root, "data") + "，以后重装还能接着用。\n" +
                                   "选「否」：全部删除。", Name, MessageBoxButtons.YesNoCancel, MessageBoxIcon.Question);
        if (keep == DialogResult.Cancel)
            return 0;
        KillOwned(root);

        foreach (string dir in new[] {
            Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory),
            Environment.GetFolderPath(Environment.SpecialFolder.Programs) })
        {
            try { File.Delete(Path.Combine(dir, Name + ".lnk")); } catch { }
        }
        try { Registry.CurrentUser.DeleteSubKeyTree(@"Software\Microsoft\Windows\CurrentVersion\Uninstall\TTSStudio", false); } catch { }

        // 自己正在运行，删不掉自己：写个临时脚本，等这个进程退出后再删目录
        var sb = new StringBuilder();
        sb.AppendLine("@echo off");
        sb.AppendLine("chcp 65001 >nul");
        sb.AppendLine("ping -n 3 127.0.0.1 >nul");
        if (keep == DialogResult.No)
        {
            sb.AppendLine("rmdir /s /q \"" + root + "\"");
        }
        else
        {
            sb.AppendLine("for /d %%d in (\"" + root + "\\*\") do if /i not \"%%~nxd\"==\"data\" rmdir /s /q \"%%d\"");
            sb.AppendLine("for %%f in (\"" + root + "\\*\") do del /f /q \"%%f\"");
        }
        sb.AppendLine("del /f /q \"%~f0\"");
        string tmp = Path.Combine(Path.GetTempPath(), "tts-studio-uninstall.cmd");
        File.WriteAllText(tmp, sb.ToString(), new UTF8Encoding(false));
        // 先弹完提示再启动删除脚本：对话框开着的时候这个 exe 还在运行，会删不掉
        MessageBox.Show(keep == DialogResult.Yes ? "已卸载。你的数据保留在 " + Path.Combine(root, "data") : "已卸载。",
                        Name, MessageBoxButtons.OK, MessageBoxIcon.Information);
        var psi = new ProcessStartInfo("cmd.exe", "/c \"" + tmp + "\"");
        psi.CreateNoWindow = true;
        psi.UseShellExecute = false;
        psi.WorkingDirectory = Path.GetTempPath();
        Process.Start(psi);
        return 0;
    }
}
