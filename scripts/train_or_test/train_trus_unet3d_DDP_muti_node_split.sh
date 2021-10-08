set -ex
echo "now is in $(pwd)"
cd /raid/lf/PROJECT/DLForPytorch

export NCCL_SOCKET_IFNAME=enp2s0f0
export NCCL_DEBUG=INFO
#export MASTER_ADDR=172.21.16.17
#export MASTER_PORT=15555
#export CUDA_VISIBLE_DEVICES="1,2,3"
python train.py  \
                --name trus_unet3d_testDDP_split_file_muti_node \
                --dataset_name trus \
                --model_name unet3d \
                --seed 1008 \
                --gpu_ids 0,1,2 \
                --visible_gpu 1,2,3 \
                \
                --dist_url 'tcp://172.21.16.17:15555'  --dist_backend 'nccl' \
                --world_size 2  --rank 0  --local_rank 0 \
                \
                --dataroot ./traces/datasets/prostate_daf3d_pre \
                --phase 'train' \
                --serial_batches --custom \
                --preprocess 'randomscale_randomcrop_ranomrotate_centercrop_rot90_mirror_
                gaussianNoise_GaussianBlur_BrightnessMultiplicative_contrast_simulate_gammatransform' \
                --crop_size '128,128,128' \
                --target_size '128,128,128' \
                --scale '1.,1.,1.' \
                --bright_mu 0.1 \
                --bright_sigma 0.5 \
                --elastic_alpha '0.,1000' \
                --elastic_sigma '10.,13.' \
                --shift_mu '0.,1000' \
                --shift_sigma '10.,13.' \
                --order_data  3   \
                --order_seg   0  \
                \
                --num_threads 8  \
                --batch_size  2  \
                \
                --input_nc  1  \
                --output_nc 1  \
                --conv_order  'crb'  \
                --init_channel_number  32 \
                --reduction 'mean'  --smooth 0 \
                --init_type  'normal'   --init_gain  1.414   --init_std 0.02 \
                \
                --optimizer_name 'adam' \
                --lr 5e-4 \
                --weight_decay 0 \
                --momentum 0.9 \
                --beta1 0.9  \
                --lr_policy 'step'  \
                --decay_epochs 60 \
                --decay_rate 0.5 \
                --warmup_lr 1e-7 \
                --warmup_epochs  30 \
                \
                --logs_dir ./traces/logs \
                --checkpoints_dir  ./traces/checkpoints  \
                --weight_path   ./traces/checkpoints/trus_unet3d_testDP1/100_net_trus_unet3d_testDP1.pth   \
                --verbose \
                --suffix '' --DEBUG  \
                --epoch_start 100 --num_epochs 150 \
                \
                --save_epoch_start 60 --save_epoch_freq 10 --save_iter_start 5000 --save_iter_freq 500 \
                --display_freq 24 --print_freq 1 --plot_freq 1 \
                --with_tensorboard  --save_log \
                --visdom_server 'http://172.21.16.17' --visdom_port 15556 \
                --visdom_env 'main'  --visdom_id 0  --visdom_ncols 0 --html_winsize 256 \
                --draw_model --display_histogram  --DDP --SyncBatchNorm --continue_train
echo "ending script"
# --local_rank 0
# --max_dataset_size  --DP --DDP  --up_interpolate --ignore_index  --lr_noise --continue_train --save_by_iter
# --with_html --with_visdom --play_video

# tcp://172.21.16.17:15555