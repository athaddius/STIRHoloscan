DATASETLOCATION=$1
HOLOHUBLOCATION=$2
docker run -it --rm --gpus "device=0" --net host --runtime=nvidia --ipc=host --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 -v /var/run/docker.sock:/var/run/docker.sock --mount src=$DATASETLOCATION,target=/workspace/data,type=bind --mount src=$HOLOHUBLOCATION,target=/workspace/holohub,type=bind --mount src=`pwd`,target=/workspace/stir,type=bind -w /workspace/stir stircontainer:2.1
