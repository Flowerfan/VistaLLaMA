train_file=vista_llama/train/train_qformer_patch.py
val_file=vista_llama/eval/run_inference_qformer_nextqa_patch.py
test_data=nextqa
checkpoint_dir=./checkpoints/
API_KEY=""



feature_dir=/mnt/bn/vlp-lq/ActivityNet_Feature
output_pred_name=${test_data}_preds



### model setting
model_name=qformerblipinit_patch8frames
video_token_len=128
num_video_frame=8
num_epoch=3
output_dir=$checkpoint_dir/$model_name
feature_dim=256
visual_model=clip14l
qformer_type=seq
eval_dir=/opt/tiger/evaluation/$model_name/$test_data

python3.10 -m torch.distributed.run --nproc_per_node=1 --master_port 9912 $train_file \
          --model_name_or_path ./LLaVA-7B-Lightening-v1-1 \
          --version v1 \
          --data_path video_chatgpt_training1.json \
          --video_folder $feature_dir \
          --tune_mm_mlp_adapter True \
          --mm_use_vid_start_end \
          --bf16 True \
          --output_dir $output_dir \
          --num_train_epochs $num_epoch \
          --qformer_type $qformer_type \
          --per_device_train_batch_size 4 \
          --per_device_eval_batch_size 4 \
          --gradient_accumulation_steps 1 \
          --evaluation_strategy "no" \
          --save_strategy "steps" \
          --save_steps 3000 \
          --save_total_limit 3 \
          --learning_rate 2e-5 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --use_visual_mask True \
          --video_token_len $video_token_len \
          --lr_scheduler_type "cosine" \
          --logging_steps 100 \
          --tf32 True \
          --model_max_length 2048 \
          --gradient_checkpointing True \
          --lazy_preprocess True

# # python3.10 $val_file \
python3.10 -m torch.distributed.run --nproc_per_node=8 --master_port 11222 $val_file \
            --video_dir /opt/tiger/NExTVideo \
            --gt_file /mnt/bn/vlp-v6/VideoLLM/nextqa/val.json \
            --video_token_len $video_token_len \
            --output_dir $output_dir \
            --output_name $output_pred_name \
            --num_video_frame $num_video_frame \
            --visual_model $visual_model \
            --qformer_type $qformer_type \
            --feature_dim $feature_dim \
            --projection_path $output_dir/mm_projector.bin \
            --use_visual_mask \
            --model-name LLaVA-7B-Lightening-v1-1 

# python3.10 quantitative_evaluation/evaluate_nextqa_gpt0613.py \
python3.10 quantitative_evaluation/evaluate_nextqa.py \
            --pred_path $checkpoints_dir/$model_name/$output_pred_name.json \
            --output_dir ${eval_dir} \
            --num_task 10 \
            --api_key  $API_KEY\
            --output_json ${output_pred_name}.json 