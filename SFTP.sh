#!/bin/bash

echo "load SFTP"
spawn sftp tieos@muse-login.hpc-lr.univ-montp2.fr
expect "password:"
sleep 1
send "Lj2ieqEvSm\r"