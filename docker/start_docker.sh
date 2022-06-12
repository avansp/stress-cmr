#!/bin/bash
set -e
set -u
set -o pipefail

CODE_DIR=`realpath $1`
DATA_DIR=`realpath $2`
IMAGE_NAME=$3

CONTAINER_NAME=stress-cmr
GPUS="--gpus all"
MEM="--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864"
USER="--user $(id -u):$(id -g)"
WD="--workdir=/app"
HOME="/app"

ACTION="docker run -itd ${GPUS} ${MEM} ${USER} ${WD} --name ${CONTAINER_NAME} -v${CODE_DIR}:/app/codes -v${DATA_DIR}:/app/data --env HOME=${HOME} ${IMAGE_NAME}" 

#set -x
echo $ACTION
eval $ACTION
