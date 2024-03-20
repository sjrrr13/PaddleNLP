# export PYTHONPATH=/root/paddlejob/workspace/env_run/xuchang/PaddleNLP/:$PYTHONPATH


# CUDA_VISIBLE_DEVICES=7 python predict_generation.py \
python -m paddle.distributed.launch --log_dir "./logs/sps/7_13" --gpus "0,1,2,3,4,5,6,7" predict_generation.py \
    --target_name_or_path meta-llama/Llama-2-13b \
    --draft_name_or_path meta-llama/Llama-2-7b \
    --src_length 512 \
    --tgt_length 256 \
    --temperature 1 \
    --top_k 0 \
    --top_p 0 \
    --data_file './HumanEval_Solution.jsonl' \
    --predict_file './outputs/sps_7b_13b.json' \
    --batch_size 1
