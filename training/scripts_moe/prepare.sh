python ./scripts/build_gsm8k_dataset.py

python scripts/moe_convertor.py \
  --input-path TODO/LLaDA2.0-mini \
  --output-path TODO/merged4moe/LLaDA2.0-mini \
  --mode merge