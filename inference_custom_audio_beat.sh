## for 25+ FPS on A100
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# OMP_NUM_THREADS=10 CUDA_VISIBLE_DEVICES=0 python -u runner.py \
#     --dataset_name beat \
#     --name beat_GesExpr_unify_addHubert_encodeHubert_mlpIncludeX_condRes_LN \
#     --n_poses 34 \
#     --ddim \
#     --ckpt fgd_best.tar \
#     --ddim \
#     --timestep_respacing ddim25 \
#     --overlap_len 4 \
#     --mode test_custom_audio \
#     --test_audio_path audios/Forrest.wav

## For 30+ FPS on 3090; For 55+ FPS on A100
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=10 CUDA_VISIBLE_DEVICES=0 python -u runner.py \
    --dataset_name beat \
    --name beat_GesExpr_unify_addHubert_encodeHubert_mlpIncludeX_condRes_LN \
    --n_poses 34 \
    --ddim \
    --ckpt fgd_best.tar \
    --ddim \
    --timestep_respacing ddim25 \
    --overlap_len 4 \
    --mode test_custom_audio \
    --jump_n_sample 2 \
    --test_audio_path audios/Forrest.wav

 