import os
import math
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import CLIPVisionModel, CLIPImageProcessor
from vista_llama.model.eva_vit import create_eva_vit_g
import torch.distributed as dist
from dataloader import KVWriter


def load_video(vis_path, num_frm=100):
    # vr = VideoReader(vis_path, ctx=cpu(0))
    vr = VideoReader(vis_path)
    total_frame_num = len(vr)
    total_num_frm = min(total_frame_num, num_frm)
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    img_array = vr.get_batch(frame_idx).numpy()  # (n_clips*num_frm, H, W, 3)

    a, H, W, _ = img_array.shape
    h, w = 224, 224
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(total_num_frm):
        clip_imgs.append(Image.fromarray(img_array[0, j]))

    return clip_imgs


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq


def get_spatio_temporal_features(features, num_temporal_tokens=100):
    t, s, c = features.shape

    temporal_tokens = np.mean(features, axis=1)
    padding_size = num_temporal_tokens - t
    if padding_size > 0:
        temporal_tokens = np.pad(temporal_tokens, ((0, padding_size), (0, 0)), mode='constant')

    spatial_tokens = np.mean(features, axis=0)
    sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0)

    return sp_features


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--video_dir_path", default="/opt/tiger/activitynet_videos", help="Path to read the videos from.")
    parser.add_argument("--clip_feat_path", default="/opt/tiger/activity_evaclip_spatio_temporal_patch", help="The output dir to save the features in.")
    parser.add_argument("--infer_batch", required=False, type=int, default=32,
                        help="Number of frames/images to perform batch inference.")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    video_dir_path = args.video_dir_path
    clip_feat_path = args.clip_feat_path
    infer_batch = args.infer_batch
    os.makedirs(clip_feat_path, exist_ok=True)

    rank = int(os.environ.get('RANK',0))
    num_gpus = torch.cuda.device_count()
    # dist.init_process_group(backend='nccl')

    output_dir = f'{args.clip_feat_path}.{rank}'
    writer = KVWriter(output_dir, 1)

    # Initialize the CLIP model
    image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16)

    vision_tower = create_eva_vit_g(224, 0, False, "fp16").to(rank)
    vision_tower.eval()

    all_videos = sorted(os.listdir(video_dir_path))
    all_videos = all_videos[rank::num_gpus]
    video_clip_features = {}
    counter = 0
    for video_name in tqdm(all_videos):
        video_path = f"{video_dir_path}/{video_name}"
        video_id = video_name.split('.')[0]
        if os.path.exists(f"{clip_feat_path}/{video_id}.pkl"):  # Check if the file is already processed
            continue
        try:
            video = load_video(video_path)
            video_tensor = image_processor.preprocess(video, return_tensors='pt')['pixel_values']
            video_tensor = video_tensor.half()

            n_chunk = len(video_tensor)
            video_features = torch.FloatTensor(n_chunk, 257, 1408).fill_(0)
            n_iter = int(math.ceil(n_chunk / float(infer_batch)))
            for i in range(n_iter):
                min_ind = i * infer_batch
                max_ind = (i + 1) * infer_batch
                video_batch = video_tensor[min_ind:max_ind].to(rank)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    image_forward_outs = vision_tower(video_batch)

                video_features[min_ind:max_ind] = image_forward_outs.detach().cpu()

            video_features = video_features.numpy().astype("float16")
            writer.write_many([video_id], [pickle.dumps(video_features)])

            counter += 1

        except Exception as e:
            print(f"Can't process {video_path}, error: {e}")

    writer.flush()


if __name__ == "__main__":
    main()
