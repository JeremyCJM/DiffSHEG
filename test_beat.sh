PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=10 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u runner.py \
    --dataset_name beat \
    --name beat_GesExpr_unify_addHubert_encodeHubert_mlpIncludeX_condRes_LN \
    --batch_size 6 \
    --n_poses 34 \
    --multiprocessing-distributed \
    --dist-url 'tcp://127.0.0.1:9935' \
    --ddim \
    --ckpt fgd_best.tar \
    --mode test_arbitrary_len \
    --ddim \
    --timestep_respacing ddim25 \
    --overlap_len 4
