#!/bin/bash

name='pengwa_development'

if [[ $(docker ps -f "name=$name" -f "status=exited" --format '{{.Names}}') == $name ]]; then
    echo "Restore Hero [$name], then exec to it"
    sudo docker start $name
    sudo docker exec -it $name /bin/bash
elif [[ $(docker ps -a -f "name=$name" -f "status=running" --format '{{.Names}}') == $name ]]; then
    echo "Hero [$name] already exists, exec to it"
    sudo docker exec -it $name /bin/bash
else
    echo "Starting container named $name"
    sudo docker run --restart always  --gpus all -v /home/pengwa/containers:/containers --net=host --shm-size=256g --ulimit memlock=-1 --ulimit stack=67108864 --name $name -it ptebic.azurecr.io/public/aifx/acpt/stable-ubuntu2004-cu121-py38-torch221 /bin/bash
