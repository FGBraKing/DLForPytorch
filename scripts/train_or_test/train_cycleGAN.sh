set -ex
echo "now is in $(pwd)"
cd /home/users/lf/CODE/PycharmProjects/DLForPytorch
python train.py --dataroot /home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/datasets/BraTs2018-IPML \
                --checkpoints_dir  /home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/checkpoints  \
                --name BraTS_cyclegan_default \
                --gpu_ids 0,1,2  \
                --model cycle_gan  \
                --input_nc 1 --output_nc 1 \
                --ngf 64 --ndf 64  \
                --netD basic \
                --netG resnet_9blocks \
                --norm instance \
                --dataset_mode 'BraTS'  \
                --num_threads 8  \
                --batch_size 15  \
                --load_size 256 --crop_size 256 \
                --preprocess None \
                --html_winsize 256 \
                --display_freq 400 --visdom_ncols 4 --visdom_id 1  \
                --visdom_server "http://127.0.0.1"  --visdom_env main --visdom_port 8097 \
                --update_html_freq 1000 --print_freq 100 \
                --save_latest_freq 5000 --save_epoch_freq 5 \
                --phase train \
                --n_epochs 200 --n_epochs_decay 200 \
                --beta1 0.5 --lr 1e-3 \
                --gan_mode lsgan \
                --pool_size 50 \
                --lr_policy linear --lr_decay_iters 50 \
                --seed 1008 \
                --verbose  \
                --normalize \
                --no_dropout \
                --continue_train \
                --load_iter 0 \
                --epoch latest \
                --epoch_count 275 \
                --data_phase 'train'




# --suffix " "


#/data/project_data_lf/DLForPytorch/datasets/BraTs2018-IPML \
