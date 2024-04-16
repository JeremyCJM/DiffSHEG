## For 50+ FPS on A100
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=10 CUDA_VISIBLE_DEVICES=0 python -u runner.py \
    --dataset_name talkshow \
    --name talkshow_GesExpr_unify_addHubert_encodeHubert_mdlpIncludeX_condRes_LN_ClsFree \
    --n_poses 88 \
    --model_base transformer_encoder \
    --classifier_free \
    --cond_scale 1.15 \
    --ckpt ckpt_e2599.tar \
    --ddim \
    --timestep_respacing ddim25 \
    --overlap_len 10 \
    --mode test_custom_audio \
    --test_audio_path audios/Forrest.wav


## For 120+ FPS on A100
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# OMP_NUM_THREADS=10 CUDA_VISIBLE_DEVICES=0 python -u runner.py \
#     --dataset_name talkshow \
#     --name talkshow_GesExpr_unify_addHubert_encodeHubert_mdlpIncludeX_condRes_LN_ClsFree \
#     --n_poses 88 \
#     --model_base transformer_encoder \
#     --classifier_free \
#     --cond_scale 1.15 \
#     --ckpt ckpt_e2599.tar \
#     --ddim \
#     --timestep_respacing ddim25 \
#     --overlap_len 10 \
#     --mode test_custom_audio \
#     --jump_n_sample 2 \
#     --test_audio_path audios/Forrest.wav