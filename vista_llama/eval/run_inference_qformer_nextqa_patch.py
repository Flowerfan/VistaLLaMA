import os
import argparse
import json
import torch
from tqdm import tqdm
from vista_llama.model import VistaLLaMAQformerLlamaForCausalLM
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from vista_llama.utils import disable_torch_init, save_result
from vista_llama.eval.model_utils import load_video
from vista_llama.video_conversation import conv_templates, SeparatorStyle
from vista_llama.model.utils import KeywordsStoppingCriteria
from vista_llama.model.dist_utils import init_distributed_mode
from vista_llama.constants import *
from vista_llama.model.eva_vit import create_eva_vit_g

DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--dataset", default="nextqa")
    parser.add_argument("--video_dir", default="/opt/tiger/NExTVideo")
    parser.add_argument(
        "--gt_file", default="/opt/tiger/SeViLA/sevila_data/nextqa/val.json"
    )
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--video_token_len", default=128, type=int)
    parser.add_argument("--use_visual_mask", action="store_true")
    parser.add_argument("--num_video_frame", default=4, type=int, help="num of used video frame")
    parser.add_argument("--output_name", default="vista_llama_qformer_nextqa_preds")
    parser.add_argument("--model-name", type=str, default="LLaVA-7B-Lightening-v1-1")
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--visual_model", type=str, default="clip14l")
    parser.add_argument("--qformer_type", type=str, default="normal")
    parser.add_argument(
        "--conv-mode", type=str, required=False, default="video-chatgpt_v1"
    )
    parser.add_argument(
        "--projection_path",
        type=str,
        default="/mnt/bn/vlp-v6/VideoLLM/qformer/mm_projector/checkpoint-9000.bin",
    )

    return parser.parse_args()


def vista_llama_infer(
    video_frames,
    question,
    conv_mode,
    model,
    vision_tower,
    tokenizer,
    image_processor,
    video_token_len,
    feature_dim,
    visual_model,
):
    """
    Run inference using the Vista-LLaMA model.

    Parameters:
    sample : Initial sample
    video_frames (torch.Tensor): Video frames to process.
    question (str): The question string.
    conv_mode: Conversation mode.
    model: The pretrained Vista-LLaMA model.
    vision_tower: Vision model to extract video features.
    tokenizer: Tokenizer for the model.
    image_processor: Image processor to preprocess video frames.
    video_token_len (int): The length of video tokens.

    Returns:
    dict: Dictionary containing the model's output.
    """

    # Prepare question string for the model
    if model.get_model().vision_config.use_vid_start_end:
        qs = (
            question
            + "\n"
            + DEFAULT_VID_START_TOKEN
            + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
            + DEFAULT_VID_END_TOKEN
        )
    else:
        qs = question + "\n" + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len

    # Prepare conversation prompt
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize the prompt
    inputs = tokenizer([prompt])

    # Preprocess video frames and get image tensor
    image_tensor = image_processor.preprocess(video_frames, return_tensors="pt")[
        "pixel_values"
    ]

    # Move image tensor to GPU and reduce precision to half
    image_tensor = image_tensor.half().to(model.device)

    # Generate video spatio-temporal features
    with torch.no_grad():
        if visual_model == "evaclip":
            with torch.cuda.amp.autocast():
                image_forward_outs = vision_tower(image_tensor)
                feature = image_forward_outs
        else:
            image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
            feature = image_forward_outs.hidden_states[-2][
                :, -feature_dim:
            ]  # Use second to last layer as in LLaVA
        video_spatio_temporal_features = feature
        # interval = feature.shape[0] // 16
        # idx = interval // 2
        # video_spatio_temporal_features = feature[idx::interval][:16]

    # Move inputs to GPU
    input_ids = torch.as_tensor(inputs.input_ids).to(model.device)

    # Define stopping criteria for generation
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # Run model inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria],
        )

    # Check if output is the same as input
    n_diff_input_output = (
        (input_ids != output_ids[:, : input_ids.shape[1]]).sum().item()
    )
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )

    # Decode output tokens
    outputs = tokenizer.batch_decode(
        output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )[0]

    # Clean output string
    outputs = outputs.strip().rstrip(stop_str).strip()

    return outputs


def initialize_model(
    model_name, projection_path=None, device=0, visual_model="clip14l"
):
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
    model = VistaLLaMAQformerLlamaForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True
    )

    # Load image processor
    image_processor = CLIPImageProcessor.from_pretrained(
        model.config.mm_vision_tower, torch_dtype=torch.float16
    )

    # Set to use start and end tokens for video
    mm_use_vid_start_end = True

    # Add tokens to tokenizer
    tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
    if mm_use_vid_start_end:
        tokenizer.add_tokens(
            [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True
        )

    # Resize token embeddings of the model
    model.resize_token_embeddings(len(tokenizer))

    # Load the weights from projection_path after resizing the token_embeddings
    if projection_path:
        print(f"Loading weights from {projection_path}")
        status = model.load_state_dict(
            torch.load(projection_path, map_location="cpu"), strict=False
        )
        if status.unexpected_keys:
            print(
                f"Unexpected Keys: {status.unexpected_keys}.\nThe Vista-LLaMA weights are not loaded correctly."
            )
        print(f"Weights loaded from {projection_path}")

    # Set model to evaluation mode and move to GPU
    model = model.eval()
    model = model.to(device)

    vision_tower_name = "openai/clip-vit-large-patch14"

    if visual_model == "clip14l":
        # Load vision tower and move to GPU
        vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device)
    elif visual_model == "evaclip":
        vision_tower = create_eva_vit_g(224, 0, False, "fp16").to(device)
    else:
        raise NotImplementedError
    vision_tower = vision_tower.eval()

    # Configure vision model
    vision_config = model.get_model().vision_config
    vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_VIDEO_PATCH_TOKEN]
    )[0]
    vision_config.use_vid_start_end = mm_use_vid_start_end
    if mm_use_vid_start_end:
        (
            vision_config.vid_start_token,
            vision_config.vid_end_token,
        ) = tokenizer.convert_tokens_to_ids(
            [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN]
        )

    # Set video token length

    return model, vision_tower, tokenizer, image_processor


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Vista-LLaMA model.

    Args:
        args: Command-line arguments.
    """
    init_distributed_mode(args)
    # Initialize the model
    video_token_len = args.video_token_len
    model, vision_tower, tokenizer, image_processor = initialize_model(
        args.model_name, args.projection_path, args.rank, args.visual_model
    )
    model.config.use_visual_mask = args.use_visual_mask
    model.config.qformer_type = args.qformer_type
    # Load both ground truth file containing questions and answers
    with open(args.gt_file) as file:
        gt_questions = json.load(file)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode

    video_formats = [".mp4", ".avi", ".mov", ".mkv"]

    # Iterate over each sample in the ground truth file
    index = 0
    # gt_questions = sorted(gt_questions)
    gt_questions = gt_questions[args.rank :: args.world_size]
    for sample in tqdm(gt_questions):
        video_name = sample["video"]
        question = sample["question"]
        id = sample["qid"]
        if args.dataset == 'nextqa':
            answer = sample["a" + str(sample["answer"])]
        else:
            answer = sample["answer"]

        sample_set = {"id": id, "question": question, "answer": answer}

        # Load the video file
        for fmt in video_formats:  # Added this line
            if args.dataset == 'nextqa':
                temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            elif args.dataset == 'activitynet_qa':
                temp_path = os.path.join(args.video_dir, f"v_{video_name}{fmt}")
            else:
                raise NotImplementedError
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        # Check if the video exists
        if os.path.exists(video_path):
            # num_frm = args.video_token_len // 32
            num_frm = args.num_video_frame
            video_frames = load_video(video_path, num_frm=num_frm)

        try:
            # Run inference on the video and add the output to the list
            output = vista_llama_infer(
                video_frames,
                question,
                conv_mode,
                model,
                vision_tower,
                tokenizer,
                image_processor,
                video_token_len,
                args.feature_dim,
                args.visual_model,
            )
            sample_set["pred"] = output
            output_list.append(sample_set)
        except Exception as e:
            print(f"Error processing video file '{video_name}': {e}")

    save_result(output_list, args.output_dir, args.output_name)
    # Save the output list to a JSON file
    # with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
    #     json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
