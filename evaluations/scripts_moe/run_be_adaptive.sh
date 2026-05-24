export CUDA_VISIBLE_DEVICES=0,1,2,3


datasets=("gsm8k" "math500" "mmlu" "arc-c")
seq_lens=(2048)
steps=(32)
block_sizes=(32)

# --evaluation-only \ 
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
                    --be-adaptive \
                    --eos-early-stop \
                    --model-name "FSCCS/dMoE-16B"
            done
        done
    done
done
