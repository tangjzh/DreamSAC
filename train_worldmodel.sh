export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --num_processes 4 train_worldmodel.py \
    --exp_name ego4d_llama_ft --output_dir log_trm --seed 0 --mixed_precision bf16 \
    --vqgan_type ctx_vqgan \
    --pretrained_model_name_or_path log_vqgan/2024-10-08-22:10:45-bair_tokenizer_ft/checkpoint-70000/unwrapped_model \
    --config_name configs/llama/config.json --load_internal_llm --language_conditioned --text_tokenizer "gpt2" \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 --lr_scheduler_type cosine \
    --oxe_data_mixes_type ego4d --resolution 64 --dataloader_num_workers 16 \
    --video_stepsize 1 --segment_length 16 --context_length 1 \
    --use_eval_dataset --use_fvd --use_frame_metrics \
    --weight_decay 0.01 --llama_attn_drop 0.1 --embed_no_wd \
    --pretrained_transformer_path pretrained_models/ivideogpt-oxe-64-act-free/transformer \