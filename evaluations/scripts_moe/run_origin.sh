export CUDA_VISIBLE_DEVICES=0,1,2,3


datasets=("gsm8k" "math500" "arc-c" "mmlu")
seq_lens=(2048)
steps=(32)
block_sizes=(32)

# --evaluation-only
# deep100: local_home1
# deep57-62: home
# hopper: scratch
for ds in "${datasets[@]}"; do
    for sl in "${seq_lens[@]}"; do
        for bs in "${block_sizes[@]}"; do
            for st in "${steps[@]}"; do
                accelerate launch --main_process_port 18100 llada_distributed_moe.py \
                    --dataset "$ds" \
                    --bsz 1 \
                    --sampling-alg low_confidence \
                    --seq-len "$sl" \
                    --block-size $bs \
                    --steps $st \
                    --temperature 0.0 \
                    --cfg 0.0 \
                    --origin \
                    --eos-early-stop \
                    --model-name "/scratch/fengsicheng/pretrained_models/LLaDA2.0-mini"
            done
        done
    done
done
