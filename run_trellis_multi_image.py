import argparse
import torch, os
torch.hub.set_dir(os.environ.get("TORCH_HUB_DIR", "/data2/hja/cache/torch/hub"))
# 可选：国内镜像与加速设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['ATTN_BACKEND'] = 'xformers' 

from glob import glob
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils


def load_images_from_dir(directory: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(directory, ext)))
    files = sorted(files)
    if len(files) == 0:
        raise FileNotFoundError(f"No images found in {directory}")
    images = [Image.open(p) for p in files]
    return images


def main():
    parser = argparse.ArgumentParser(description='Run TRELLIS multi-image 3D reconstruction; infer format from output_path extension')
    parser.add_argument('--input_dir', type=str, required=True, help='directory containing multi-view images')
    parser.add_argument('--output_path', type=str, default='./tmp/trellis_multi.ply', help='output path (.ply or .obj or .glb)')
    parser.add_argument('--model_dir', type=str, default='/data2/hja/CKPT/TRELLIS/TRELLIS-image-large', help='model directory')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()

    images = load_images_from_dir(args.input_dir)

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model_dir)
    pipeline.cuda()

    outputs = pipeline.run_multi_image(
        images,
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
            fill_holes=False,
        )
        glb.export(args.output_path)
        print(f"[TRELLIS] Saved {ext} to {args.output_path}")
    else:
        raise ValueError(f"Unsupported output extension '.{ext}'. Use .ply (point cloud) or .obj or .glb.")


if __name__ == '__main__':
    main()
