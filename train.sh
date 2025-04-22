accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 train.py \
    --model VQ-16 \
    --data-path /home/azureuser/v-yuqianhong/ImageNet/ILSVRC2012/train \
    --image-size 256 \
    --global-batch-size 128 \
    "$@"
