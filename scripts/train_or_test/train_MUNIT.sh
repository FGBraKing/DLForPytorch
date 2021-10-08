set -ex
echo "now is in $(pwd)"
cd /home/users/lf/CODE/PycharmProjects/DLForPytorch
python MUNIT_train.py  --dataroot /home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/datasets/BraTs2018-IPML \
                       --name BraTS_MUNIT_default \
                       --checkpoints_dir  /home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/checkpoints  \
                       --gpu_ids  0,1  \
                --model 'MUNIT'  \
                --dataset_mode 'BraTS'  \
                --phase 'train' \
                --num_threads 8  \
                --batch_size 2  \
                --load_size 256 --crop_size 256 \
                --max_dataset_size 999999999 \
                --preprocess None \
                --html_winsize 256 \
                --display_freq 400 --display_ncols 8 --display_id 1  \
                --visdom_server "http://172.21.141.4"  --display_env main --display_port 8097 \
                --update_html_freq 1000 --print_freq 100 \
                --save_latest_freq 5000 --save_epoch_freq 10 \
                --n_epochs 100 --n_epochs_decay 200 \
                --seed 1008 \
                --verbose  \
                --normalize \
                --load_iter 0 \
                --epoch latest \
                --epoch_count 1 \
                --config '/home/users/lf/CODE/PycharmProjects/DLForPytorch/configs/defaults/BraTs_MUNIT_train.yaml'  \
                --log_dir  '/home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/logs' \
                --lr_policy linear --lr_decay_iters 60 \
                --lr 1e-3

              # --serial_batches  --no_flip --normalize --suffix ''   --no_html  --save_by_iter  --continue_train
              # --pool_size 50 \  --gamma 0.1  --resume
              # --lr_policy linear --lr_decay_iters 60 \
#                              --beta1 0.5 --lr 1e-3 \
#                --gan_mode lsgan \                       --no_dropout \
