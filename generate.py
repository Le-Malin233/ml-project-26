import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from sketch_rnn.model import SketchRNN, sample_unconditional
from sketch_rnn.hparams import hparam_parser

# ---------- stroke utilities ----------

def strokes_to_stroke5(x, v):
    """
    Convert (x, v) to stroke-5 format.
    x: [N, 2]  -> dx, dy
    v: [N]     -> pen state {0,1,2}
    return: [N, 5] numpy array
    """
    N = x.size(0)
    stroke5 = torch.zeros(N, 5, device=x.device)
    stroke5[:, :2] = x
    # pen state one-hot: p1,p2,p3
    # v==0 -> p1=1, v==1 -> p2=1, v==2 -> p3=1
    stroke5[torch.arange(N), v + 2] = 1.0
    return stroke5

def render_stroke5_to_png(stroke5, save_path, linewidth=2, dpi=150):
    """
    Render stroke-5 to PNG.
    stroke5: (N,5) numpy array
    """
    x, y = 0.0, 0.0
    xs, ys = [], []

    plt.figure(figsize=(4, 4))
    plt.axis('equal')
    plt.axis('off')

    for dx, dy, p_down, p_up, p_end in stroke5:
        x += dx
        y += dy

        if p_down == 1:
            xs.append(x)
            ys.append(y)
        else:
            if len(xs) > 1:
                plt.plot(xs, ys, 'k', linewidth=linewidth)
            xs, ys = [], []

        if p_end == 1:
            break

    if len(xs) > 1:
        plt.plot(xs, ys, 'k', linewidth=linewidth)

    # invert y-axis to match QuickDraw convention
    plt.gca().invert_yaxis()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

# ---------- generation ----------

@torch.no_grad()
def generate_many(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model
    model = SketchRNN(args)

    # load checkpoint
    ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)

    model.to(device).eval()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Using device: {device}')
    print(f'Generating {args.num_samples} sketches...')

    for i in range(args.num_samples):
        # unconditional sampling
        x, v = sample_unconditional(
            model,
            T=args.temperature,
            z_scale=args.z_scale,
            device=device
        )

        # convert to 5-d stroke
        stroke5 = strokes_to_stroke5(x, v).cpu().numpy()  # <-- 输出5维

        # save npy (5维)
        npy_path = os.path.join(args.output_dir, f'sketch_{i:05d}.npy')
        np.save(npy_path, stroke5)

        # save png
        png_path = os.path.join(args.output_dir, f'sketch_{i:05d}.png')
        render_stroke5_to_png(stroke5, png_path)

        if (i + 1) % 10 == 0 or i == 0:
            print(f'  {i+1}/{args.num_samples} done')

    print('Generation finished.')

# ---------- main ----------

if __name__ == '__main__':
    hp = hparam_parser()
    parser = argparse.ArgumentParser(parents=[hp])

    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='generated')
    parser.add_argument('--temperature', type=float, default=0.65)
    parser.add_argument('--z_scale', type=float, default=1.0)

    args = parser.parse_args()
    generate_many(args)
