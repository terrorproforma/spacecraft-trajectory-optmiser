$KEY = "C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\traj-key.pem"
$HOST_ = "ubuntu@192.222.55.229"
$SSHOPTS = @("-i", $KEY, "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20")
function rsh { param([string]$cmd) & ssh @SSHOPTS $HOST_ $cmd }
function rput { param([string]$local, [string]$remote) & scp @SSHOPTS $local "$HOST_`:$remote" }
function rget { param([string]$remote, [string]$local) & scp @SSHOPTS "$HOST_`:$remote" $local }
function wlf { param([string]$path, [string]$text) [IO.File]::WriteAllText($path, ($text -replace "`r`n", "`n")) }
# copy local script C:\Users\Angus\h100work\s\<name> to ~/s/<name> and run in foreground
function rrun { param([string]$name) rput "C:\Users\Angus\h100work\s\$name" "s/$name"; rsh "bash ~/s/$name" }
# run in background with nohup, log at ~/logs/<name>.log
function rbg { param([string]$name) rput "C:\Users\Angus\h100work\s\$name" "s/$name"; rsh "nohup bash ~/s/$name > ~/logs/$name.log 2>&1 < /dev/null & echo started pid=`$!" }
function wrun { param([string]$name) & wsl.exe -e bash -lc "sed -i 's/\r$//' /mnt/c/Users/Angus/h100work/s/$name && bash /mnt/c/Users/Angus/h100work/s/$name" }