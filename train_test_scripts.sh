################################# BEAT #################################

### Train on BEAT.
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=10 CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -u runner.py \
    --dataset_name beat \
    --name beat_diffsheg \
    --batch_size 2500 \
    --num_epochs 1000 \
    --save_every_e 20 \
    --eval_every_e 40 \
    --n_poses 34 \
    --ddim \
    --multiprocessing-distributed \
    --dist-url 'tcp://127.0.0.1:6666'
         


### Test on BEAT.
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=10 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u runner.py \
    --dataset_name beat \
    --name beat_GesExpr_unify_addHubert_encodeHubert_mlpIncludeX_condRes_LN \
    --n_poses 34 \
    --multiprocessing-distributed \
    --dist-url 'tcp://127.0.0.1:8888' \
    --ckpt fgd_best.tar \
    --mode test_arbitrary_len \
    --ddim \
    --timestep_respacing ddim25 \
    --overlap_len 4


################################# SHOW #################################

### Train on SHOW.
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=10 CUDA_VISIBLE_DEVICES=1,2,3,4,5 python -u runner.py \
    --dataset_name talkshow \
    --name talkshow_diffsheg \
    --batch_size 950 \
    --num_epochs 4000 \
    --save_every_e 20 \
    --eval_every_e 40 \
    --n_poses 88 \
    --classifier_free \
    --multiprocessing-distributed \
    --dist-url 'tcp://127.0.0.1:6667' \
    --ddim \
    --max_eval_samples 200
    


### Test on SHOW.
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=10 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u runner.py \
    --dataset_name talkshow \
    --name talkshow_GesExpr_unify_addHubert_encodeHubert_mdlpIncludeX_condRes_LN_ClsFree \
    --PE pe_sinu \
    --n_poses 88 \
    --multiprocessing-distributed \
    --dist-url 'tcp://127.0.0.1:8889' \
    --classifier_free \
    --cond_scale 1.25 \
    --ckpt ckpt_e2599.tar \
    --mode test_arbitrary_len \
    --ddim \
    --timestep_respacing ddim25 \
    --overlap_len 10
    