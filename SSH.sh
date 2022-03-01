#!/usr/bin/expect -f

echo "load SSH"
spawn ssh tieos@muse-login.hpc-lr.univ-montp2.fr
expect "password:"
sleep 1
send "Lj2ieqEvSm\r"
interact