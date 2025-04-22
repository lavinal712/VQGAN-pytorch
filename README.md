# VQGAN

## Training

```bash
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 train.py \
    --model VQ-16 \
    --data-path /path/to/ImageNet/train \
    --image-size 256 \
    --global-batch-size 128 \
```

## Acknowledgements

- [LlamaGen](https://github.com/FoundationVision/LlamaGen)
