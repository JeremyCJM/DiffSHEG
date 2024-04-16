PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=10 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u runner.py \
    --dataset_name talkshow \
    --name talkshow_GesExpr_unify_addHubert_encodeHubert_mdlpIncludeX_condRes_LN_ClsFree \
    --batch_size 1 \
    --PE pe_sinu \
    --n_poses 88 \
    --model_base transformer_encoder \
    --multiprocessing-distributed \
    --dist-backend 'nccl' \
    --dist-url 'tcp://127.0.0.1:3915' \
    --classifier_free \
    --cond_scale 1.25 \
    --ckpt ckpt_e2599.tar \
    --mode test_arbitrary_len \
    --ddim \
    --timestep_respacing ddim25 \
    --overlap_len 10