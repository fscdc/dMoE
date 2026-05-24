PYTHONPATH=$(pwd)/VeOmni:$PYTHONPATH sh train.sh tasks/train_llada2_bd.py configs/moe/llada2_mini_adaptive.yaml


# python scripts/moe_convertor.py \
#   --input-path ./logs/llada2_mini_dmoe/checkpoints/global_step_TODO/hf_ckpt/ \
#   --output-path TODO/llada2_mini_dmoe/ \
#   --mode split