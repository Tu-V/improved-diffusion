export OPENAI_LOGDIR="simple-shapes-5k-checkpoints"
rm -rf $OPENAI_LOGDIR
MODEL_FLAGS="--image_size 64 --num_channels 64 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --lr_anneal_steps 125000"

python scripts/image_train.py --data_dir simple-shapes-5k $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
# NUM_GPUS=4
# mpiexec -n $NUM_GPUS python scripts/image_train.py --data_dir simple-shapes-5k $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS


#sampling
export OPENAI_LOGDIR="simple-shapes-5k-checkpoints-samping"
rm -rf $OPENAI_LOGDIR
MODEL_FLAGS="--image_size 64 --num_channels 64 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --lr_anneal_steps 125000"

