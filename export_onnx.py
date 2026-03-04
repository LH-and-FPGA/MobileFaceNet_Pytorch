"""
导出模型为 ONNX 格式
用法:
    # 从完整模型 (.pth) 导出
    python export_onnx.py --resume model/068.pth

    # 从 checkpoint (.ckpt) 导出
    python export_onnx.py --resume model/068.ckpt

    # 指定输出路径
    python export_onnx.py --resume model/068.pth --output mobilefacenet.onnx
"""

import argparse
import torch
import importlib
from config import MODEL_FILE, MODEL_SIZE


def load_model(resume):
    if resume.endswith('.pth'):
        net = torch.load(resume, map_location='cpu')
        print(f'完整模型加载完成: {resume}')
    else:
        model_module = importlib.import_module(f'core.{MODEL_FILE}')
        if MODEL_SIZE == 'tiny':
            net = model_module.MobileFacenet()
        elif MODEL_SIZE == 'small':
            net = model_module.MobileFacenet(
                model_module.Mobilefacenet_small_setting, inplanes=32, mid_channels=256)
        else:
            net = model_module.MobileFacenet(
                model_module.Mobilefacenet_bottleneck_setting, inplanes=64, mid_channels=512)
        ckpt = torch.load(resume, map_location='cpu')
        net.load_state_dict(ckpt['net_state_dict'])
        print(f'Checkpoint 加载完成: {resume}  (MODEL_FILE={MODEL_FILE}, MODEL_SIZE={MODEL_SIZE})')
    net.eval()
    return net


def main():
    parser = argparse.ArgumentParser(description='导出 ONNX 模型')
    parser.add_argument('--resume', type=str, required=True, help='模型路径 (.pth 或 .ckpt)')
    parser.add_argument('--output', type=str, default=None, help='输出 ONNX 文件路径')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.resume.rsplit('.', 1)[0] + '.onnx'

    net = load_model(args.resume)

    # 输入: batch x 3 x 112 x 96
    dummy_input = torch.randn(1, 3, 112, 96)

    torch.onnx.export(
        net,
        dummy_input,
        args.output,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'},
        },
        opset_version=11,
    )
    print(f'ONNX 模型已导出: {args.output}')


if __name__ == '__main__':
    main()
