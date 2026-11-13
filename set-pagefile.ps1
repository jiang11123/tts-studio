# 把虚拟内存（页面文件）调大到足以加载 Breeze-TTS-2
#
# 背景：加载 6.5 GB 权重要一次性提交约 8 GB。页面文件原来由系统自动管理（约 17 GB），
# 加 16 GB 物理内存，提交上限只有 32~35 GB；日常程序（Chrome / Claude / WorkBuddy）
# 已经把提交额度吃到 28 GB，剩 4~7 GB 不够 → OSError 1455，
# 严重时直接 0xc0000005 崩掉整个网页服务。
#
# 授权方式：右键「以管理员身份运行」，或双击后点 UAC 的「是」。
# 改完必须重启才生效。

$ErrorActionPreference = 'Stop'

# 全程留日志：窗口一闪就关也能查到底跑到哪一步、报了什么错
try {
    Start-Transcript -Path (Join-Path $PSScriptRoot 'pagefile-setup.log') -Append -Force | Out-Null
} catch { }

function Say($t) { Write-Host $t }

# ---- 1/6 备份现状 ----
$stamp  = Get-Date -Format 'yyyyMMdd-HHmmss'
$backup = Join-Path $PSScriptRoot "pagefile-backup-$stamp.txt"
$cs = Get-WmiObject Win32_ComputerSystem
$lines = @(
    "备份时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    "系统自动管理页面文件: $($cs.AutomaticManagedPagefile)",
    "---- 页面文件设置 ----"
)
Get-WmiObject Win32_PageFileSetting |
    ForEach-Object { $lines += "$($_.Name)  初始 $($_.InitialSize) MB / 最大 $($_.MaximumSize) MB" }
Get-WmiObject Win32_PageFileUsage |
    ForEach-Object { $lines += "使用中: $($_.Name)  已分配 $($_.AllocatedBaseSize) MB  峰值 $($_.PeakUsage) MB" }
$lines | Out-File -FilePath $backup -Encoding utf8 -Force
Say "1/6 现状已备份到：$backup"

# ---- 2/6 磁盘空间 ----
$d = Get-WmiObject Win32_LogicalDisk -Filter "DeviceID='D:'"
if (-not $d) { throw "找不到 D 盘" }
$dFreeGB = [math]::Round($d.FreeSpace / 1GB, 1)
if ($dFreeGB -lt 32) { throw "D 盘只剩 $dFreeGB GB，放不下 28 GB，先清理再运行" }
Say "2/6 D 盘剩余 $dFreeGB GB，计划放 28 GB"

# ---- 3/6 关掉「自动管理」 ----
if ($cs.AutomaticManagedPagefile) {
    $cs.AutomaticManagedPagefile = $false
    $cs.Put() | Out-Null
    Say "3/6 已关闭「自动管理所有驱动器的分页文件大小」"
} else {
    Say "3/6 系统原本就不是自动管理，跳过"
}

# ---- 4/6 先建 D 盘 ----
# 顺序很关键：先把 D 盘建好并回读确认，再动 C 盘。
# 上一版是先改 C 再建 D，结果 C 那步报错直接中断，两边都没落成。
Say "4/6 创建 D 盘页面文件 28672 MB ..."
Get-WmiObject Win32_PageFileSetting |
    Where-Object { $_.Name -like 'D:*' } |
    ForEach-Object { try { $_.Delete() } catch { } }
Set-WmiInstance -Class Win32_PageFileSetting -Arguments @{
    Name        = 'D:\pagefile.sys'
    InitialSize = 28672
    MaximumSize = 28672
} | Out-Null
$dNow = Get-WmiObject Win32_PageFileSetting | Where-Object { $_.Name -like 'D:*' }
if (-not $dNow) { throw "D 盘页面文件没建出来（WMI 回读为空）" }
if ([int]$dNow.MaximumSize -ne 28672) {
    throw "D 盘页面文件写进去了但值不对：回读 MaximumSize=$($dNow.MaximumSize)，期望 28672"
}
Say "    D: -> $($dNow.InitialSize) MB ~ $($dNow.MaximumSize) MB  已确认"

# ---- 5/6 再把 C 盘压到 4 GB ----
# C 盘只剩 30 多 G，不能让系统托管的分页文件继续吃它。
# 这一步失败不影响 D 盘，最多是 C 盘维持「系统管理大小」。
Say "5/6 把 C 盘压到 4096 MB ..."
try {
    $cNow = Get-WmiObject Win32_PageFileSetting | Where-Object { $_.Name -like 'C:*' } | Select-Object -First 1
    if ($cNow) {
        $cNow.InitialSize = 4096
        $cNow.MaximumSize = 4096
        $cNow.Put() | Out-Null
    } else {
        Set-WmiInstance -Class Win32_PageFileSetting -Arguments @{
            Name = 'C:\pagefile.sys'; InitialSize = 4096; MaximumSize = 4096
        } | Out-Null
    }
    Say "    C: -> 4096 MB  已确认"
} catch {
    Say "    C 盘没改成功（不影响 D 盘已生效）：$($_.Exception.Message)"
    Say "    C 盘会维持「系统管理大小」，重启后照常能开机，只是仍占 C 盘空间。"
}

# ---- 6/6 回读 ----
Say ""
Say "6/6 现在的配置："
$sum = 0
Get-WmiObject Win32_PageFileSetting | ForEach-Object {
    Say ("    {0}  {1} MB ~ {2} MB" -f $_.Name, $_.InitialSize, $_.MaximumSize)
    $sum += [int]$_.MaximumSize
}
Say ""
Say "页面文件合计 $([math]::Round($sum/1024,1)) GB；重启后提交上限约 16(物理) + 该值"
Say "备份文件：$backup"
Say "改完必须重启电脑才生效。"

try { Stop-Transcript | Out-Null } catch { }
