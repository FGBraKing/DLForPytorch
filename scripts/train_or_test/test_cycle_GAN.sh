set -ex
echo "now is in $(pwd)"
cd /home/users/lf/CODE/PycharmProjects/DLForPytorch
python test.py --dataroot /home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/datasets/BraTs2018-IPML \
               --checkpoints_dir  /home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/checkpoints  \
               --name BraTS_cyclegan_default \
               --gpu_ids 0 \
               --model cycle_gan  \
               --input_nc 1  --output_nc 1 \
               --ngf 64 --ndf 64  \
               --netD basic --netG resnet_9blocks \
               --norm instance  \
               --dataset_mode 'BraTS'  \
               --num_threads 8  \
               --batch_size 1  \
               --load_size 256 --crop_size 256 \
               --preprocess None \
               --display_winsize 256 \
               --phase test \
               --results_dir /home/users/lf/CODE/PycharmProjects/DLForPytorch/traces/results  \
               --aspect_ratio 1.0  \
               --num_test 200 \
               --no_dropout \
               --seed 1008 \
               --verbose  \
               --epoch 330

