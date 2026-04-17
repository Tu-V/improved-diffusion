export OPENAI_LOGDIR="simple-shapes-5k-16x16-output/num_channel_64/checkpoints"

MODEL_FLAGS="--image_size 16 --num_channels 64 --num_res_blocks 3 --attention_resolutions 8"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --lr_anneal_steps 200000"

python scripts/image_train.py --data_dir simple-shapes-5k-16x16 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
