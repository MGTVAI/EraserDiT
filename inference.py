import argparse
from utils.common import test_time,GlobalValues
from utils.pre import VideoInpaintPre
from utils.inference_utils import init, inference_batch
from utils.post import post_stream_normalized
from utils.post_pkg import FFmpegWriter
import os
import math
import datetime
import decord
import ffmpeg
import torch


GlobalValues.DEBUG=False
output_dir = "results"
negative_prompt = "Colorful color tone, overexposure, static, blurry details, subtitles, style, artwork, picture, static, overall graying, worst quality, low-quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, finger fusion, still image, cluttered background, three legs, many people in the background, walking backwards, no noise"


@test_time(enable=GlobalValues.ENABLE_PER)
def run_batch(video_path,video_mask_path,bbox_path,prompt, vis_flag,
              model_path="jieeliu/EraserDiT", cpu_offload=False, vae_tiling=False):
    print(f"video: {video_path}")
    print(f"mask: {video_mask_path}")
    print(f"prompt: {prompt}")

    org_video_info = ffmpeg.probe(video_path)
    height_input_ori = int(org_video_info["streams"][0]["height"])
    width_input_ori = int(org_video_info["streams"][0]["width"])
    bit_rate = int(org_video_info["streams"][0]["bit_rate"])//1000000
    crop_flag = True if height_input_ori * width_input_ori >1088 * 1920 else False

    vid_name = os.path.basename(video_path).split(".mp4")[0]
    sub_output_dir = os.path.join(output_dir, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-")+vid_name)
    save_path = f"{sub_output_dir}/{vid_name}_results_final_crop_{crop_flag}.mp4"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # init
    device=torch.device("cuda")
    weight_dtype=torch.bfloat16
    seed=None

    preprocessor = VideoInpaintPre(device=device,
                                   align_h=32,
                                   align_w=32,
                                   ksize=(9,9),
                                   dilate_iter=9,
                                   shift_alpha=1*8+1,
                                   TEMP_INFER_LEN=121,
                                   crop_flag=crop_flag,
                                   )
    if crop_flag:
        output_bbox = preprocessor.bbox_video(height_input_ori, width_input_ori, bbox_path)
        if output_bbox is None:
            return "The maximum mask movement in the current video sequence exceeds 1080 × 1920."
    else:
        output_bbox = None

    # init inference
    pipeline = init(device, weight_dtype, pre_dir=model_path,
                    cpu_offload=cpu_offload, vae_tiling=vae_tiling)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    pre_video_shift = None
    video_save_writer = None      # type: FFmpegWriter

    current_batch=0
    while True:
        video_ori, mask_ori, fps, videos_input_ori, masks_input_ori = preprocessor.load_videos(video_path=video_path,
                                                            mask_path=video_mask_path,
                                                            bbox_path=bbox_path,
                                                            # decord_device=decord.gpu(0),
                                                            decord_device=decord.cpu(0),
                                                            sample_rate=1,
                                                            batch_idx=current_batch)
        
        if video_ori is None:
            # video to end
            break

        video_input, mask_input, _ = preprocessor(video_ori,mask_ori,batch_idx=current_batch,format="nhwc")
        input_shape = preprocessor.TranslateShape(video_input.shape,src="nchw",dst="nhwc")

        if video_save_writer is None:
            if crop_flag:
                video_save_writer = FFmpegWriter(path=save_path,width=videos_input_ori.shape[2],height=videos_input_ori.shape[1],fps=fps, bitrate="{}M".format(bit_rate))
            else:
                video_save_writer = FFmpegWriter(path=save_path,width=video_ori.shape[2],height=video_ori.shape[1],fps=fps, bitrate="{}M".format(bit_rate))

        # init mask shift
        if current_batch==0:
            # masks_zero_shift = torch.zeros((math.ceil(preprocessor.shift_alpha/8),mask_input.shape[1],mask_input.shape[2],mask_input.shape[3]),device=device,dtype=mask_input.dtype)
            masks_zero_shift = torch.zeros((math.ceil(preprocessor.shift_alpha/8),mask_input.shape[1],mask_input.shape[2],mask_input.shape[3]),dtype=mask_input.dtype)   # (2,1,736,1280)
        else:
            video_input=torch.cat([pre_video_shift,video_input])
            mask_input=torch.cat([masks_zero_shift,mask_input])     

        print(f"batch-{current_batch}: len={video_ori.shape[0]}.")
        output_frames = inference_batch(videos=video_input,masks_input=mask_input,prompt=prompt,negative_prompt=negative_prompt,
                                        pipeline=pipeline,generator=generator,
                                        device=device,weight_dtype=weight_dtype)
        pre_video_shift = output_frames[-preprocessor.shift_alpha:,...].cpu()
        
        if current_batch==0:
            post_stream_normalized(output_frames=output_frames,ori_shape=video_ori.shape, model_video_shape=input_shape,writer=video_save_writer,
                                   crop_flag=crop_flag, videos_input_ori=videos_input_ori,
                                   video_ori=video_ori, mask_ori=mask_ori, output_bbox=output_bbox,
                                   write_to=True)
        else:
            post_stream_normalized(output_frames=output_frames[preprocessor.shift_alpha:,...],ori_shape=video_ori.shape, model_video_shape=input_shape,writer=video_save_writer,
                                   crop_flag=crop_flag, videos_input_ori=videos_input_ori,
                                   video_ori=video_ori, mask_ori=mask_ori, output_bbox=output_bbox,
                                   write_to=True)
        current_batch+=1
    
    if video_save_writer is not None:
        video_save_writer.Close()
        video_save_writer=None
    return save_path

def main():
    # surpport ltx 095
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_path", type=str, default="data/10268234.mp4", help="Input video path.") # 
    parser.add_argument("--mask_path", type=str, default="data/10268234_mask.mp4", help="Input mask path.") #
    # parser.add_argument("--vid_path", type=str, default="data/113000356.mp4", help="Input video path.") # 
    # parser.add_argument("--mask_path", type=str, default="data/113000356_mask.mp4", help="Input mask path.") #
    parser.add_argument("--bbox_path", type=str, default=None, help="Input mask path.If the resolution is less than 1080p, the bbox_path can be set to None.") # 
    parser.add_argument("--prompt", type=str, default="There is a bridge over the lake.", help="A brief description of the video content") # 
    parser.add_argument("--model_path", default="jieeliu/EraserDiT", help="Local model directory or Hugging Face model ID.")
    parser.add_argument("--cpu_offload", action="store_true", help="Move inactive stage models to CPU.")
    parser.add_argument("--vae_tiling", action="store_true", help="Use spatial VAE tiling to reduce activation memory.")

    args = parser.parse_args()
    
    output = run_batch(args.vid_path, args.mask_path, args.bbox_path, args.prompt, vis_flag=False,
                       model_path=args.model_path, cpu_offload=args.cpu_offload, vae_tiling=args.vae_tiling)

    print(output)

if __name__=='__main__':
    main()
