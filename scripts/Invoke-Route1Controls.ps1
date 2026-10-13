[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$OutputDirectory,[switch]$Pilot,[switch]$Resume)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$routeScriptRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
$routeProjectRoot=[System.IO.Path]::GetFullPath((Join-Path $routeScriptRoot '..'))
Set-Location -LiteralPath $routeProjectRoot
. (Join-Path $routeScriptRoot 'Set-ProjectEnvironment.ps1') | Out-Null
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUNBUFFERED='1'
$env:PYTHONDONTWRITEBYTECODE='1'
foreach($routeKey in @('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')){[Environment]::SetEnvironmentVariable($routeKey,'4','Process')}
$routeOut=[System.IO.Path]::GetFullPath((Join-Path $routeProjectRoot $OutputDirectory))
if(-not $routeOut.StartsWith((Join-Path $routeProjectRoot 'runs\quick')+'\',[System.StringComparison]::OrdinalIgnoreCase)){throw 'Output outside runs/quick'}
if((Test-Path -LiteralPath $routeOut) -and -not $Resume){throw 'Output already exists'}
$routeMonitor=$routeOut+'-monitor-'+(Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
New-Item -ItemType Directory -Path $routeMonitor | Out-Null
$routeSummary=[ordered]@{status='initializing';started_at_utc=(Get-Date).ToUniversalTime().ToString('o');child_pid=$null;error=$null;device='cpu';monitor=$routeMonitor}
$routeChild=$null
$routeLock=Join-Path $routeProjectRoot 'runs\.route1-controls.lock.json'
$routeToken=[guid]::NewGuid().ToString('N')
$routeOwned=$false
try{
 $routeOs=Get-CimInstance Win32_OperatingSystem
 if($routeOs.FreePhysicalMemory*1KB/1GB -lt 3.25 -or $routeOs.FreeVirtualMemory*1KB/1GB -lt 3.25){throw 'CPU admission needs 3.25 GiB free RAM and commit'}
 if((Get-PSDrive E).Free/1GB -lt 10){throw 'Need 10 GiB disk reserve'}
 $routeStream=[System.IO.File]::Open($routeLock,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
 $routeWriter=[System.IO.StreamWriter]::new($routeStream,[System.Text.UTF8Encoding]::new($false))
 try{$routeWriter.Write(([ordered]@{pid=$PID;token=$routeToken;output=$routeOut}|ConvertTo-Json))}finally{$routeWriter.Dispose()}
 $routeOwned=$true
 $routeEntry=if($Resume){'scripts/resume_route1_controls.py'}else{'scripts/run_route1_controls.py'}
 $routeArgs=@('-B',$routeEntry,'--output-dir',$OutputDirectory)
 if($Pilot){$routeArgs+='--pilot'}
 if($Resume){$routeArgs+='--resume'}
 $routeLauncher=Start-Process -FilePath (Join-Path $routeProjectRoot '.venv\Scripts\python.exe') -ArgumentList $routeArgs -WorkingDirectory $routeProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $routeMonitor 'stdout.log') -RedirectStandardError (Join-Path $routeMonitor 'stderr.log') -PassThru
 $routeChild=$routeLauncher
 Start-Sleep -Milliseconds 300
 $routeDesc=@(Get-CimInstance Win32_Process -Filter ('ParentProcessId = {0}' -f $routeLauncher.Id)|Where-Object{$_.ExecutablePath -like "$routeProjectRoot\*" -and $_.Name -eq 'python.exe'})
 if($routeDesc.Count -eq 1){$routeChild=Get-Process -Id $routeDesc[0].ProcessId}
 $routeSummary.child_pid=$routeChild.Id;$routeSummary.status='running'
 while(-not $routeLauncher.HasExited){
  $routeChild.Refresh()
  if($routeChild.HasExited){break}
  $routeOs=Get-CimInstance Win32_OperatingSystem
  try{
   $routeCpu=$routeChild.TotalProcessorTime
   if($null -eq $routeCpu){$routeChild.Refresh();if($routeChild.HasExited){break};throw 'CPU counter unavailable'}
   $routeSample=[ordered]@{time=(Get-Date).ToUniversalTime().ToString('o');pid=$routeChild.Id;working_set_bytes=$routeChild.WorkingSet64;peak_working_set_bytes=$routeChild.PeakWorkingSet64;private_commit_bytes=$routeChild.PrivateMemorySize64;free_ram_gib=$routeOs.FreePhysicalMemory*1KB/1GB;free_commit_gib=$routeOs.FreeVirtualMemory*1KB/1GB;cpu_seconds=$routeCpu.TotalSeconds;device='cpu'}
  }catch{$routeChild.Refresh();if($routeChild.HasExited){break};throw}
  [System.IO.File]::AppendAllText((Join-Path $routeMonitor 'resources.jsonl'),($routeSample|ConvertTo-Json -Compress)+[Environment]::NewLine,[System.Text.UTF8Encoding]::new($false))
  if(-not $routeChild.HasExited -and ($routeSample.working_set_bytes/1GB -gt 2.5 -or $routeSample.free_ram_gib -lt 1.5 -or $routeSample.free_commit_gib -lt 2)){$routeChild.Kill();$routeChild.WaitForExit(10000)|Out-Null;throw 'Owned CPU process stopped at resource reserve boundary'}
  Start-Sleep -Seconds 2
  $routeLauncher.Refresh()
 }
 $routeLauncher.WaitForExit();$routeSummary['exit_code']=$routeLauncher.ExitCode
 $routeSummary.status=if($routeLauncher.ExitCode -eq 0){'completed'}else{'failed'}
}catch{$routeSummary.status='failed';$routeSummary.error=$_.Exception.Message}
finally{
 if($null -ne $routeChild){$routeChild.Refresh();if(-not $routeChild.HasExited){$routeChild.Kill();$routeChild.WaitForExit(10000)|Out-Null}}
 $routeSummary['finished_at_utc']=(Get-Date).ToUniversalTime().ToString('o')
 [System.IO.File]::WriteAllText((Join-Path $routeMonitor 'summary.json'),($routeSummary|ConvertTo-Json -Depth 5),[System.Text.UTF8Encoding]::new($false))
 if($routeOwned -and (Test-Path -LiteralPath $routeLock)){$routeExisting=Get-Content -LiteralPath $routeLock -Raw|ConvertFrom-Json;if($routeExisting.token -ceq $routeToken){[System.IO.File]::Delete($routeLock)}}
}
$routeSummary|ConvertTo-Json -Depth 5
if($routeSummary.status -ne 'completed'){exit 1}
