set -ex
echo "now is in $(pwd)"
#export MASTER_ADDR=172.21.16.17
#export MASTER_PORT=15554
#export WORLD_SIZE=6
#export RANK=0
#export LOCAL_RANK=2
#export CUDA_VISIBLE_DEVICES="0,1,2,3"
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export NCCL_SOCKET_IFNAME=enp2s0f0

cd /raid/lf/PROJECT/DLForPytorch
python  train_optimized.py    \
                      --name trus_unet3d_testDDP_TEST_forDDP2 \
                      --dataset_name trus \
                      --model_name unet3d \
                      --seed 1008 \
                      --gpu_ids 0,1,2,3 \
                      --visible_gpu 0,1,2,3 \
                      \
                      --dist_url 'tcp://172.21.16.17:15000'  --dist_backend 'nccl' \
                      --world_size 4  --rank 0 \
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
                      --batch_size  6  \
                      \
                      --input_nc  1  \
                      --output_nc 1  \
                      --conv_order  'crb'  \
                      --init_channel_number  32 \
                      --reduction 'mean'  --smooth 0 \
                      --init_type  'kaiming'   --init_gain  1.414   --init_std 0.02 \
                      \
                      --optimizer_name 'adam' \
                      --lr 1e-3 \
                      --weight_decay 0 \
                      --momentum 0.9 \
                      --beta1 0.9  \
                      --lr_policy 'step'  \
                      --decay_epochs 60 \
                      --decay_rate 0.5 \
                      --warmup_lr 2e-7 \
                      --warmup_epochs  30 \
                      \
                      --logs_dir ./traces/logs \
                      --checkpoints_dir  ./traces/checkpoints  \
                      --weight_path  ./traces/checkpoints/trus_unet3d_testDDP_TEST_forDDP/11_net_trus_unet3d_testDDP_TEST_forDDP.pth  \
                      --verbose \
                      --suffix '' --DEBUG  \
                      --epoch_start 11 --num_epochs 100 \
                      \
                      --save_epoch_start 1 --save_epoch_freq 10 --save_iter_start 5000 --save_iter_freq 500 \
                      --display_freq 72 --print_freq 1 --plot_freq 1 \
                      --with_tensorboard  --save_log \
                      --visdom_server 'http://172.21.16.17' --visdom_port 15556 \
                      --visdom_env 'main'  --visdom_id 0  --visdom_ncols 0 --html_winsize 256 \
                      --draw_model   --SyncBatchNorm  --continue_train  --DDP

# --local_rank 0 此时local_rank可有可无，因为会自动设置
# --max_dataset_size  --DP --DDP  --up_interpolate --ignore_index  --lr_noise --continue_train -save_by_iter
# --with_html --with_visdom --play_video  --display_histogram tcp://172.21.16.17:15567 --world_size -1  --DDP