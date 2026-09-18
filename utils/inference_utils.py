from utils.common import GlobalValues
from utils.common import test_time
import torch
from typing import Tuple

from transformers import T5EncoderModel, T5Tokenizer
from pipelines.pipeline_ltx_video2video import LTXVideoToVideoPipeline
from diffusers import FlowMatchEulerDiscreteScheduler
from models import AutoencoderKLLTXVideo
from models import LTXVideoTransformer3DModel


@test_time(enable=GlobalValues.ENABLE_PER)
def init(device, weight_dtype, pre_dir="jieeliu/EraserDiT"):
    base_model_path = f"{pre_dir}"
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_path, subfolder="scheduler")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        base_model_path, subfolder="vae", use_safetensor=True, low_cpu_mem_usage=False
    )

    text_encoder = T5EncoderModel.from_pretrained(base_model_path, subfolder="text_encoder", revision="main", variant=None,
                                                    torch_dtype=torch.bfloat16)
    tokenizer = T5Tokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    ltx_model = LTXVideoTransformer3DModel.from_pretrained(
        base_model_path, subfolder="transformer", use_safetensor=True, low_cpu_mem_usage=False
    )
    
    vae.eval()
    text_encoder.eval()
    ltx_model.eval()

    pipeline = LTXVideoToVideoPipeline(
        vae=vae, 
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=ltx_model, 
        scheduler=noise_scheduler,
    )

    pipeline.to(device, weight_dtype)
    pipeline.set_progress_bar_config(disable=False)
    return pipeline

@test_time(enable=GlobalValues.ENABLE_PER)
def inference_batch(videos: torch.Tensor, masks_input: torch.Tensor, prompt=None, negative_prompt=None,pipeline=None,generator=None,device=None,weight_dtype=torch.float32)->Tuple[torch.Tensor]:
    with torch.no_grad():
        with torch.autocast(str(device), dtype=weight_dtype):
            frame_normalized = pipeline(video = videos,                             # current batch
                                        masks = masks_input,                         # current batch
                                        prompt=prompt,
                                        negative_prompt=negative_prompt,
                                        num_frames=videos.shape[0],
                                        height=videos.shape[2],
                                        width=videos.shape[3],
                                        num_inference_steps=50,
                                        generator=generator, 
                                        output_type="pt", 
                                        strength=0.8,
                                        # device=accelerator.device,
                                        decode_timestep=0.0, # 0.03
                                        decode_noise_scale=0.0,  # 0.025
                                    ).frames[0]
            return frame_normalized
