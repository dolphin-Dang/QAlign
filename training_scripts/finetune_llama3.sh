PROJECT_PATH=/home/dangrf/QAlign
DATASET=gsmtrans_gsm8kinstruct_question_all-en
# DATASET=metamath_all
# MODEL_PATH=/home/nfs04/dangrf/model/llama-3-8b-QAlign
MODEL_PATH=/home/nfs02/model/llama2/hf/Llama-2-7b-hf
OUTPUT_PATH=/home/nfs04/dangrf/model/lora-adapters/llama-2-7b-QAlign

export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=1234 \
    finetune.py \
	--num_train_epochs 3 \
    --learning_rate 2e-5 \
    --data_path "$PROJECT_PATH/data/$DATASET" \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_PATH" \
    --overwrite_output_dir \
    --bf16 true \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters false \
    --do_train \
    --lr_scheduler_type "cosine" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --save_strategy "no" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --load_best_model_at_end true \
    --logging_steps 10 \
    --seed 42
