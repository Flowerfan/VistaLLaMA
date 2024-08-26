import os
import argparse
import json
import torch
from tqdm import tqdm
from vista_llama.model import VideoChatGPTQformerLlamaForCausalLM
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from vista_llama.utils import disable_torch_init, save_result
from vista_llama.eval.model_utils import load_video
from vista_llama.inference import video_chatgpt_infer
import torch.distributed as dist
from vista_llama.constants import *
from vista_llama.model.dist_utils import init_distributed_mode

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', default='/opt/tiger/NExTVideo')
    parser.add_argument('--gt_file', default='/opt/tiger/SeViLA/sevila_data/nextqa/val.json')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--output_name', default='video_chatgpt_qformer_nextqa_preds')
    parser.add_argument("--model-name", type=str, default='LLaVA-7B-Lightening-v1-1')
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--projection_path", type=str, default='/mnt/bn/vlp-v6/Video-ChatGPT/qformer/mm_projector/checkpoint-9000.bin')

    return parser.parse_args()

def initialize_model(model_name, projection_path=None, device=0):
    """
    Initializes the model with given parameters.

    Parameters:
    model_name (str): Name of the model to initialize.
    projection_path (str, optional): Path to the projection weights. Defaults to None.

    Returns:
    tuple: Model, vision tower, tokenizer, image processor, vision config, and video token length.
    """

    # Disable initial torch operations
    disable_torch_init()

    # Convert model name to user path
    model_name = os.path.expanduser(model_name)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    model = VideoChatGPTQformerLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                         use_cache=True)

    # Load image processor
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

    # Set to use start and end tokens for video
    mm_use_vid_start_end = True

    # Add tokens to tokenizer
    tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
    if mm_use_vid_start_end:
        tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)

    # Resize token embeddings of the model
    model.resize_token_embeddings(len(tokenizer))

    # Load the weights from projection_path after resizing the token_embeddings
    if projection_path:
        print(f"Loading weights from {projection_path}")
        status = model.load_state_dict(torch.load(projection_path, map_location='cpu'), strict=False)
        if status.unexpected_keys:
            print(f"Unexpected Keys: {status.unexpected_keys}.\nThe Video-ChatGPT weights are not loaded correctly.")
        print(f"Weights loaded from {projection_path}")

    # Set model to evaluation mode and move to GPU
    model = model.eval()
    model = model.to(device)

    vision_tower_name = "openai/clip-vit-large-patch14"

    # Load vision tower and move to GPU
    vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name, torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True).to(device)
    vision_tower = vision_tower.eval()

    # Configure vision model
    vision_config = model.get_model().vision_config
    vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]
    vision_config.use_vid_start_end = mm_use_vid_start_end
    if mm_use_vid_start_end:
        vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])

    # Set video token length
    video_token_len = 32

    return model, vision_tower, tokenizer, image_processor, video_token_len

def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    init_distributed_mode(args)
    # rank = int(os.environ.get('RANK',0))
    # num_gpus = torch.cuda.device_count()
    # dist.init_process_group(backend='nccl')

    # Initialize the model
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name, args.projection_path, args.rank)
    # Load both ground truth file containing questions and answers
    with open(args.gt_file) as file:
        gt_questions = json.load(file)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    gt_questions = sorted(gt_questions)
    gt_questions = gt_questions[args.rank::args.world_size]
    for sample in tqdm(gt_questions):
        video_name = sample['video']
        question = sample['question']
        id = sample['qid']
        answer = sample['a' + str(sample['answer'])]

        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        # Check if the video exists
        if os.path.exists(video_path):
            video_frames = load_video(video_path)

        try:
            # Run inference on the video and add the output to the list
            output = video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len)
            sample_set['pred'] = output
            output_list.append(sample_set)
        except Exception as e:
            print(f"Error processing video file '{video_name}': {e}")

    # Save the output list to a JSON file
    save_result(output_list, args.output_dir, args.output_name)

    # with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
    #     json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
