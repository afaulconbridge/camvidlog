#!/bin/bash
set -e

# read secrets into env
if [ -f .devcontainer/.secrets.env ]; then
    set -a
    . .devcontainer/.secrets.env
    set +a
fi

sudo mkdir -p /mnt/shared
sudo mount -t cifs -o rw,vers=3.0,user=adam,pass=${CIFS_PASS},dir_mode=0775,file_mode=0775,uid=1000,gid=9999 //192.168.1.102/shared-mirror /mnt/shared
