import argparse
import torch, os
torch.hub.set_dir(os.environ.get("TORCH_HUB_DIR", "/data2/hja/cache/torch/hub"))
# 可选：国内镜像与加速设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['ATTN_BACKEND'] = 'xformers' 
# 如需固定GPU：
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils


def main():
    parser = argparse.ArgumentParser(description='Run TRELLIS image-to-3D; infer output format from output_path extension')
    parser.add_argument('--input_path', type=str, required=True, help='input image path')
    parser.add_argument('--output_path', type=str, default='./tmp/trellis_image.ply', help='output path (.ply or .obj or .glb)')
    parser.add_argument('--model_dir', type=str, default='/data2/hja/CKPT/TRELLIS/TRELLIS-image-large', help='model directory')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()

    image = Image.open(args.input_path)

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model_dir)
    pipeline.cuda()

    outputs = pipeline.run(
        image,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # 从输出文件扩展名推断格式
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
