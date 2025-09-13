set -x
algo=mpc_mappo
# task=Platform/PlatformHover
# algo=mappo
task=FormationGateTraversal
current_datetime=$(date +'%Y-%m-%d-%H-%M-%S')
echo logs/res-$current_datetime.log  # Made consistent with the redirect below
TORCH_LOGS=not_implemented python -u train.py task=$task algo=$algo > logs/res-$current_datetime.log 2>&1