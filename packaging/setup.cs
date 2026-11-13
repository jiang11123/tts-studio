// TTS Studio 安装包：把内嵌的 payload.zip 解压到用户目录，建快捷方式，登记到「设置 → 应用」里能卸载。
// 装在 %LOCALAPPDATA%\Programs 下，不需要管理员权限。用 Windows 自带的 csc.exe 编译（C# 5 语法）。
using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Win32;

[assembly: AssemblyTitle("TTS Studio 安装程序")]
[assembly: AssemblyProduct("TTS Studio")]

class SetupForm : Form
{
    const string Name_ = "TTS Studio";
    static readonly Color Accent = Color.FromArgb(124, 92, 255);
    TextBox pathBox;
    CheckBox deskBox, runBox;
    Button goBtn, browseBtn;
    ProgressBar bar;
    Label status, title, sub;
    string version;
    bool finished, launchAfter;
    bool shortcuts = true, register = true;
    string error;
    string installedExe, installedDir;

    [STAThread]
    static int Main(string[] args)
    {
        // 静默安装（测试 / 批量部署用）：/S [/D=目录] [/NOSHORTCUTS] [/NOREG]
        if (Array.IndexOf(args, "/S") >= 0)
        {
            string dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Programs\TTS Studio");
            foreach (string a in args) if (a.StartsWith("/D=")) dir = a.Substring(3);
            var f = new SetupForm();
            f.shortcuts = Array.IndexOf(args, "/NOSHORTCUTS") < 0;
            f.register = Array.IndexOf(args, "/NOREG") < 0;
            f.Install(dir, f.shortcuts, false);
            if (f.error != null) { Console.Error.WriteLine(f.error); return 1; }
            return 0;
        }
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.Run(new SetupForm());
        return 0;
    }

    SetupForm()
    {
        version = ReadVersion();
        Text = Name_ + " " + version + " 安装";
        Font = new Font("Microsoft YaHei UI", 9.5f);
        FormBorderStyle = FormBorderStyle.FixedDialog;
        MaximizeBox = false;
        StartPosition = FormStartPosition.CenterScreen;
        ClientSize = new Size(560, 330);
        BackColor = Color.White;
        try { Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath); } catch { }

        title = new Label { Text = "安装 TTS Studio", Font = new Font("Microsoft YaHei UI", 16f, FontStyle.Bold),
                            ForeColor = Color.FromArgb(29, 29, 39), AutoSize = true, Location = new Point(28, 24) };
        sub = new Label { Text = "本地多模型配音软件 · 版本 " + version + "\n安装包不带模型和 PyTorch：第一次打开时自动联网装运行环境，模型在软件「设置」里下载。",
                          ForeColor = Color.FromArgb(107, 107, 122), AutoSize = false, Size = new Size(510, 44),
                          Location = new Point(30, 66) };
        var pathLbl = new Label { Text = "安装位置", AutoSize = true, Location = new Point(30, 128) };
        pathBox = new TextBox { Location = new Point(30, 152), Size = new Size(410, 26),
                                Text = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                                                    @"Programs\TTS Studio") };
        browseBtn = new Button { Text = "浏览…", Location = new Point(450, 150), Size = new Size(80, 30) };
        browseBtn.Click += delegate
        {
            using (var dlg = new FolderBrowserDialog())
            {
                dlg.Description = "选择安装位置（会在里面建一个 TTS Studio 文件夹）";
                if (dlg.ShowDialog(this) == DialogResult.OK)
                    pathBox.Text = Path.Combine(dlg.SelectedPath, "TTS Studio");
            }
        };
        deskBox = new CheckBox { Text = "创建桌面快捷方式", Checked = true, AutoSize = true, Location = new Point(30, 192) };
        runBox = new CheckBox { Text = "装完马上打开", Checked = true, AutoSize = true, Location = new Point(190, 192) };
        bar = new ProgressBar { Location = new Point(30, 232), Size = new Size(500, 8), Visible = false };
        status = new Label { Text = "需要约 4 GB 硬盘空间（含之后自动安装的运行环境，模型另算）。", AutoSize = false,
                             Size = new Size(380, 40), Location = new Point(30, 252), ForeColor = Color.FromArgb(138, 138, 149) };
        goBtn = new Button { Text = "安装", Location = new Point(420, 270), Size = new Size(110, 38), FlatStyle = FlatStyle.Flat,
                             BackColor = Accent, ForeColor = Color.White, Font = new Font("Microsoft YaHei UI", 10.5f, FontStyle.Bold) };
        goBtn.FlatAppearance.BorderSize = 0;
        goBtn.Click += delegate { Start(); };
        Controls.AddRange(new Control[] { title, sub, pathLbl, pathBox, browseBtn, deskBox, runBox, bar, status, goBtn });
        AcceptButton = goBtn;
    }

    static string ReadVersion()
    {
        try
        {
            using (var zip = OpenPayload())
            {
                var e = zip.GetEntry("VERSION");
                if (e != null)
                    using (var r = new StreamReader(e.Open())) return r.ReadToEnd().Trim();
            }
        }
        catch { }
        return "";
    }

    static ZipArchive OpenPayload()
    {
        var s = Assembly.GetExecutingAssembly().GetManifestResourceStream("payload.zip");
        return new ZipArchive(s, ZipArchiveMode.Read);
    }

    void Ui(Action a) { if (IsHandleCreated) BeginInvoke(a); }

    void Start()
    {
        if (finished)  // 装完后同一个按钮变成「完成」
        {
            if (launchAfter)
                Process.Start(new ProcessStartInfo(installedExe) { WorkingDirectory = installedDir, UseShellExecute = false });
            Close();
            return;
        }
        string dir = pathBox.Text.Trim();
        if (dir.Length == 0) return;
        // 升级时旧版还开着，文件会被占用
        string dll = Path.Combine(dir, @"runtime\python\python311.dll");
        if (File.Exists(dll))
        {
            try { using (File.Open(dll, FileMode.Open, FileAccess.ReadWrite, FileShare.None)) { } }
            catch
            {
                MessageBox.Show(this, "TTS Studio 正在运行，先把它的窗口关掉再安装。", Name_, MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
        }
        goBtn.Enabled = browseBtn.Enabled = pathBox.Enabled = deskBox.Enabled = runBox.Enabled = false;
        bar.Visible = true;
        bool desk = deskBox.Checked, run = runBox.Checked;
        new Thread(delegate () { Install(dir, desk, run); }) { IsBackground = true }.Start();
    }

    void Install(string dir, bool desk, bool run)
    {
        try
        {
            Directory.CreateDirectory(dir);
            long total = 0;
            using (var zip = OpenPayload())
            {
                int n = zip.Entries.Count, i = 0;
                foreach (var e in zip.Entries)
                {
                    i++;
                    string dest = Path.GetFullPath(Path.Combine(dir, e.FullName));
                    if (!dest.StartsWith(Path.GetFullPath(dir), StringComparison.OrdinalIgnoreCase)) continue;
                    if (e.FullName.EndsWith("/")) { Directory.CreateDirectory(dest); continue; }
                    Directory.CreateDirectory(Path.GetDirectoryName(dest));
                    e.ExtractToFile(dest, true);
                    total += e.Length;
                    if (i % 40 == 0 || i == n)
                    {
                        int pct = i * 100 / n;
                        string nm = e.FullName;
                        Ui(delegate { bar.Value = pct; status.Text = "正在复制文件… " + pct + "%\n" + nm; });
                    }
                }
            }
            string exe = Path.Combine(dir, "TTS Studio.exe");
            if (shortcuts)
                Shortcut(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs), Name_ + ".lnk"), exe, dir);
            if (desk)
                Shortcut(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), Name_ + ".lnk"), exe, dir);
            if (register)
            using (var k = Registry.CurrentUser.CreateSubKey(@"Software\Microsoft\Windows\CurrentVersion\Uninstall\TTSStudio"))
            {
                k.SetValue("DisplayName", Name_);
                k.SetValue("DisplayVersion", version);
                k.SetValue("Publisher", Name_);
                k.SetValue("DisplayIcon", exe);
                k.SetValue("InstallLocation", dir);
                k.SetValue("UninstallString", "\"" + exe + "\" --uninstall");
                k.SetValue("EstimatedSize", (int)(total / 1024 + 3500000), RegistryValueKind.DWord);  // 加上首次安装的依赖约 3.5 GB
                k.SetValue("NoModify", 1, RegistryValueKind.DWord);
                k.SetValue("NoRepair", 1, RegistryValueKind.DWord);
            }
            Ui(delegate
            {
                bar.Value = 100;
                status.Text = "安装完成。" + (run ? "第一次打开会自动装运行环境，需要联网十几分钟。" : "双击桌面或开始菜单里的 TTS Studio 打开。");
                goBtn.Text = "完成";
                goBtn.Enabled = true;
                finished = true; launchAfter = run; installedExe = exe; installedDir = dir;
            });
        }
        catch (Exception ex)
        {
            error = ex.Message;
            Ui(delegate
            {
                MessageBox.Show(this, "安装失败：" + ex.Message, Name_, MessageBoxButtons.OK, MessageBoxIcon.Error);
                goBtn.Enabled = browseBtn.Enabled = pathBox.Enabled = deskBox.Enabled = runBox.Enabled = true;
                bar.Visible = false;
                status.Text = "";
            });
        }
    }

    static void Shortcut(string lnk, string target, string workdir)
    {
        Type t = Type.GetTypeFromProgID("WScript.Shell");
        object sh = Activator.CreateInstance(t);
        object sc = t.InvokeMember("CreateShortcut", BindingFlags.InvokeMethod, null, sh, new object[] { lnk });
        Type st = sc.GetType();
        st.InvokeMember("TargetPath", BindingFlags.SetProperty, null, sc, new object[] { target });
        st.InvokeMember("WorkingDirectory", BindingFlags.SetProperty, null, sc, new object[] { workdir });
        st.InvokeMember("IconLocation", BindingFlags.SetProperty, null, sc, new object[] { target + ",0" });
        st.InvokeMember("Description", BindingFlags.SetProperty, null, sc, new object[] { "TTS Studio 本地配音" });
        st.InvokeMember("Save", BindingFlags.InvokeMethod, null, sc, null);
    }
}
