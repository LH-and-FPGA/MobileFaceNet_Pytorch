"""
人脸相似度测试脚本
用法:
    python test_similarity.py --img1 img/face1.jpg --img2 img/face2.jpg
    python test_similarity.py --img1 img/face1.jpg --img2 img/face2.jpg --resume model/xxx/070.ckpt
"""

import argparse
import importlib
import numpy as np
import torch
from PIL import Image
from config import MODEL_FILE, MODEL_SIZE


def load_model(resume, device):
    # .pth 文件为完整模型，直接加载即可；.ckpt 文件需要重建模型再加载 state_dict
    if resume.endswith('.pth'):
        net = torch.load(resume, map_location=device)
        net.to(device)
        net.eval()
        print(f'完整模型加载完成: {resume}')
        return net

    model_module = importlib.import_module(f'core.{MODEL_FILE}')

    if MODEL_SIZE == 'tiny':
        net = model_module.MobileFacenet()
    elif MODEL_SIZE == 'small':
        net = model_module.MobileFacenet(
            model_module.Mobilefacenet_small_setting, inplanes=32, mid_channels=256)
    else:
        net = model_module.MobileFacenet(
            model_module.Mobilefacenet_bottleneck_setting, inplanes=64, mid_channels=512)

    ckpt = torch.load(resume, map_location=device)
    net.load_state_dict(ckpt['net_state_dict'])
    net.to(device)
    net.eval()
    print(f'模型加载完成: {resume}  (MODEL_FILE={MODEL_FILE}, MODEL_SIZE={MODEL_SIZE})')
    return net


def preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((96, 112))  # W x H
    img = np.array(img, dtype=np.float32)
    img = (img - 127.5) / 128.0
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0)


def cosine_similarity(feat1, feat2):
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))


def main():
    parser = argparse.ArgumentParser(description='人脸相似度测试')
    parser.add_argument('--img1', type=str, required=True)
    parser.add_argument('--img2', type=str, required=True)
    parser.add_argument('--resume', type=str, default='./model-2-22-1350/best/068.ckpt')
    args = parser.parse_args()

    device = torch.device('cpu')
    net = load_model(args.resume, device)

    img1 = preprocess(args.img1).to(device)
    img2 = preprocess(args.img2).to(device)

    with torch.no_grad():
        # 原图 + 水平翻转，拼接特征
        feat1 = net(img1).cpu().numpy().flatten()
        feat2 = net(img2).cpu().numpy().flatten()
        feat1_flip = net(torch.flip(img1, dims=[3])).cpu().numpy().flatten()
        feat2_flip = net(torch.flip(img2, dims=[3])).cpu().numpy().flatten()
        feat1 = np.concatenate([feat1, feat1_flip])
        feat2 = np.concatenate([feat2, feat2_flip])

    similarity = cosine_similarity(feat1, feat2)

    print('=' * 50)
    print(f'图片1: {args.img1}')
    print(f'图片2: {args.img2}')
    print(f'余弦相似度: {similarity:.4f}')
    print('=' * 50)


if __name__ == '__main__':
    main()
