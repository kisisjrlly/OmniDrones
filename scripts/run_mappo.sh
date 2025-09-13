algo=mappo
task=Platform/PlatformHover
current_datetime=$(date +'%Y-%m-%d-%H-%M-%S')
echo  logs/res.log-$current_datetime
TORCH_LOGS=not_implemented python -u train.py task=$$task algo=$$algo > logs/res-$current_datetime.log 2>&1