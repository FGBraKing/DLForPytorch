set -ex
echo "now is in $(pwd)"
#export MASTER_ADDR=localhost
#export MASTER_PORT=34567
#export CUDA_VISIBLE_DEVICES="0,1,2,4"
cd /raid/lf/PROJECT/DLForPytorch
python -m torch.distributed.launch --nproc_per_node 4 \
                train.py  \
                --name trus_unet3d_testDDP \
                                --dataset_name trus \
                                --model_name unet3d \
                                --seed 1008 \
                                --gpu_ids 0,1,2,3 \
                                \
                                --world_size 4  \
                                --dist_url 'tcp://172.21.16.17:15555'  --dist_backend 'nccl' \
                                \
                                --dataroot /raid/lf/PROJECT/DLForPytorch/traces/datasets/prostate_daf3d_pre \
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
                                --logs_dir /raid/lf/PROJECT/DLForPytorch/traces/logs \
                                --checkpoints_dir  /raid/lf/PROJECT/DLForPytorch/traces/checkpoints  \
                                --weight_path  None  \
                                --verbose \
                                --suffix '' --DEBUG  \
                                --epoch_start 1 --num_epochs 200 \
                                \
                                --save_epoch_start 150 --save_epoch_freq 10 --save_iter_start 5000 --save_iter_freq 500 \
                                --display_freq 60 --print_freq 1 --plot_freq 1 \
                                --with_tensorboard  --save_log \
                                --display_server 'http://172.21.16.17' --display_port 15556 \
                                --display_env 'main'  --display_id 0  --display_ncols 0 --display_winsize 256 \
                                --draw_model --display_histogram  --DDP

# --local_rank 0
# --max_dataset_size  --DP --DDP  --up_interpolate --ignore_index  --lr_noise --continue_train -save_by_iter
# --with_html --with_visdom --play_video