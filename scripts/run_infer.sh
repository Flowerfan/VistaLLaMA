dataset=nextqa
video_dir=data/NExTVideo/
gt_file=./nextqa/val.json
checkpoints_dir=./checkpoints


model_name=qformer_patch12frameseva257_seq_3ep
model_name=qformer_patch16frameseva257_seqmask_3ep
output_pred_name=test_eval_mask
eval_dir=./evaluation/${model_name}/${output_pred_name}
video_token_len=128
num_video_frame=8
visual_model=evaclip
# visual_model=clip14l
qformer_type=seq
feature_dim=257
val_file=vista_llama/eval/run_inference_qformer_nextqa_patch.py
api_key=API_KEY


# python -m torch.distributed.run --nproc_per_node=2 --master_port 11322 $val_file \
# CUDA_VISIBLE_DEVICES=2 python $val_file \
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_port 11022 $val_file \
            --dataset $dataset \
            --video_dir $video_dir \
            --gt_file $gt_file \
            --video_token_len $video_token_len \
            --output_dir $checkpoints_dir/$model_name \
            --num_video_frame $num_video_frame \
            --feature_dim $feature_dim \
            --output_name $output_pred_name \
            --qformer_type $qformer_type \
            --visual_model $visual_model \
            --projection_path $checkpoints_dir/$model_name/mm_projector.bin \
            --model-name LLaVA-7B-Lightening-v1-1 


python3.10 quantitative_evaluation/evaluate_nextqa.py \
            --pred_path $checkpoints_dir/$model_name/$output_pred_name.json \
            --output_dir ${eval_dir} \
            --num_task 10 \
            --api_key $API_KEY \
            --output_json ${output_pred_name}.json 
