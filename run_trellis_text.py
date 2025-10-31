import argparse
import torch, os
torch.hub.set_dir(os.environ.get("TORCH_HUB_DIR", "/data2/hja/cache/torch/hub"))
# 可选：国内镜像与加速设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['ATTN_BACKEND'] = 'xformers' 
# 如需镜像：os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import postprocessing_utils


def main():
    parser = argparse.ArgumentParser(description='Run TRELLIS text-to-3D; infer output format from output_path extension')
    parser.add_argument('--prompt', type=str, required=True, help='text prompt')
    parser.add_argument('--output_path', type=str, default='./tmp/trellis_text.ply', help='output path (.ply or .obj or .glb)')
    parser.add_argument('--model_dir', type=str, default='/data2/hja/CKPT/TRELLIS/TRELLIS-text-base', help='model directory')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()

    pipeline = TrellisTextTo3DPipeline.from_pretrained(args.model_dir)
    pipeline.cuda()

    outputs = pipeline.run(
        args.prompt,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    ext = os.path.splitext(args.output_path)[1].lower().lstrip('.')
    if ext == 'ply':
        outputs['gaussian'][0].save_ply(args.output_path)
        print(f"[TRELLIS] Saved PLY to {args.output_path}")
    elif ext == 'obj' or ext == 'glb':
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.95,
            texture_size=1024,
        )
        glb.export(args.output_path)
        print(f"[TRELLIS] Saved {ext} to {args.output_path}")
    else:
        raise ValueError(f"Unsupported output extension '.{ext}'. Use .ply (point cloud) or .obj or .glb.")


if __name__ == '__main__':
    main()
