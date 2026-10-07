"""Quick validation: model creation, weight loading, forward pass"""
import argparse
import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import delayformer_fixpooling

class Args:
    task_name = 'pretrain'
    channel_independence = 1
    d_model = 512
    pe = 'fix_pe'
    project = 'conv'
    dropout = 0.1
    n_heads = 8
    e_layers = 6
    d_ff = 2048
    seq_len = 1024
    L = 500
    n_vars = 1
    p1 = 25
    p2 = 50
    pooling_kernel = 30
    pooling_type = 'avg'
    pred_len = 96
    output_attention = False

def test_model(name, ckpt_path, n_heads, e_layers):
    print(f"\n{'='*60}")
    print(f"Testing {name} model: {ckpt_path}")
    args = Args()
    args.n_heads = n_heads
    args.e_layers = e_layers
    model = delayformer_fixpooling.Model(args)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model created! Total params: {total_params:,}")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
    
    # Only these archived tensors belong to the inactive channel-mixing head.
    inactive_shapes = {
        'mlp_head.0.weight': (96, 512), 'mlp_head.0.bias': (96,),
        'pred_dim_change.0.weight': (1, 7), 'pred_dim_change.0.bias': (1,),
        'pred_dim_change.1.weight': (1,), 'pred_dim_change.1.bias': (1,),
    }
    for key, shape in inactive_shapes.items():
        if key in state_dict:
            if tuple(state_dict[key].shape) != shape:
                raise ValueError(f'Unexpected inactive tensor shape: {key}')
            del state_dict[key]

    model.load_state_dict(state_dict, strict=True)
    print(f"  Weights loaded successfully!")

    model.eval()
    x = torch.randn(2, 1024, 1)
    x_mark = torch.zeros(2, 1024, 1)
    dec_inp = torch.zeros(2, 96, 1)
    y_mark = torch.zeros(2, 96, 1)
    with torch.no_grad():
        out = model(x, x_mark, dec_inp, y_mark)
    print(f"  Forward pass OK! Output shape: {out.shape}")
    assert out.shape == (2, 96, 1), f"Unexpected shape: {out.shape}"
    assert torch.isfinite(out).all(), 'Forecast contains non-finite values'
    print(f"  {name} model: ALL PASSED!")

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-size', choices=['small', 'medium', 'large', 'all'],
                        default='all')
    parser.add_argument('--checkpoint', help='Local checkpoint path (single model only)')
    options = parser.parse_args()
    if options.checkpoint and options.model_size == 'all':
        parser.error('--checkpoint requires a single --model-size')
    configurations = {'small': (8, 6), 'medium': (12, 10), 'large': (16, 12)}
    selected = configurations if options.model_size == 'all' else [options.model_size]
    torch.manual_seed(0)
    for name in selected:
        heads, layers = configurations[name]
        path = options.checkpoint or os.path.join(base, '12Bcheckpoints/new', name, 'checkpoint.pth')
        test_model(name.title(), path, n_heads=heads, e_layers=layers)
    print('SELECTED MODELS VALIDATED SUCCESSFULLY!')
