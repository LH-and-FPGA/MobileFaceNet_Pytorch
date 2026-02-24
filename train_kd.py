import os
import torch
import torch.utils.data
import torch.nn.functional as F
from torch import nn
from torch.nn import DataParallel
from datetime import datetime
from config import BATCH_SIZE, SAVE_FREQ, RESUME, SAVE_DIR, TEST_FREQ, TOTAL_EPOCH, MODEL_PRE, GPU
from config import CASIA_DATA_DIR, LFW_DATA_DIR, MODEL_FILE, MODEL_SIZE
import importlib
from core import model as _orig_model  # teacher always uses original PReLU model
from core.utils import init_log
from dataloader.CASIA_Face_loader import CASIA_Face
from dataloader.LFW_loader import LFW
from torch.optim import lr_scheduler
import torch.optim as optim
import time
from tqdm import tqdm
from lfw_eval import parseList, evaluation_10_fold
import numpy as np
import scipy.io

# ============================================================
# Knowledge Distillation Config
# ============================================================
TEACHER_CKPT = './model-2-22-1350/best/068.ckpt'
                    # 最好的大模型 checkpoint 路径

STUDENT_INIT = './model/CASIA_B512_v2_20260222_160714/051.ckpt'   # 用已有的小模型做热启动（从 epoch 1 重新跑完整 KD 计划）
                    # 例如：'./model/CASIA_B512_v2_20260222_160714/051.ckpt'
                    # 填这个后 RESUME 留空

# RESUME 在 config.py 里设置，用于恢复被中断的 KD 训练（接着之前的 epoch 继续跑）

KD_LAMBDA = 1.0     # KD loss 权重，可在 0.5~2.0 之间调整
# ============================================================


if __name__ == '__main__':
    assert TEACHER_CKPT, "请先设置 TEACHER_CKPT 路径"

    # gpu init
    gpu_list = ''
    multi_gpus = False
    if isinstance(GPU, int):
        gpu_list = str(GPU)
    else:
        multi_gpus = True
        for i, gpu_id in enumerate(GPU):
            gpu_list += str(gpu_id)
            if i != len(GPU) - 1:
                gpu_list += ','
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    # other init
    start_epoch = 1
    save_dir = os.path.join(SAVE_DIR, MODEL_PRE + 'kd_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info

    # define trainloader and testloader
    trainset = CASIA_Face(root=CASIA_DATA_DIR)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=8, pin_memory=True, drop_last=False)

    nl, nr, folds, flags = parseList(root=LFW_DATA_DIR)
    testdataset = LFW(nl, nr)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=32,
                                             shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    # ---------- student ----------
    model_module = importlib.import_module(f'core.{MODEL_FILE}')
    if MODEL_SIZE == 'tiny':
        student = model_module.MobileFacenet()
    elif MODEL_SIZE == 'small':
        student = model_module.MobileFacenet(model_module.Mobilefacenet_small_setting, inplanes=32, mid_channels=256)
    else:  # 'original'
        student = model_module.MobileFacenet(model_module.Mobilefacenet_bottleneck_setting, inplanes=64, mid_channels=512)
    ArcMargin = model_module.ArcMarginProduct(128, trainset.class_nums)

    # ---------- teacher (frozen, always uses core.model original) ----------
    teacher = _orig_model.MobileFacenet()
    ckpt = torch.load(TEACHER_CKPT)
    teacher.load_state_dict(ckpt['net_state_dict'])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    _print('Teacher loaded from: {}'.format(TEACHER_CKPT))

    # 热启动：加载已有小模型权重，但从 epoch 1 重新跑完整 KD 计划
    if STUDENT_INIT:
        ckpt = torch.load(STUDENT_INIT)
        student.load_state_dict(ckpt['net_state_dict'])
        _print('Student initialized from: {}'.format(STUDENT_INIT))

    # define optimizers
    ignored_params = list(map(id, student.linear1.parameters()))
    ignored_params += list(map(id, ArcMargin.weight))
    prelu_params = []
    for m in student.modules():
        if isinstance(m, nn.PReLU):
            ignored_params += list(map(id, m.parameters()))
            prelu_params += list(m.parameters())
    base_params = filter(lambda p: id(p) not in ignored_params, student.parameters())

    optimizer_groups = [
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': student.linear1.parameters(), 'weight_decay': 4e-4},
        {'params': ArcMargin.weight, 'weight_decay': 4e-4},
    ]
    if prelu_params:
        optimizer_groups.append({'params': prelu_params, 'weight_decay': 0.0})
    optimizer_ft = optim.SGD(optimizer_groups, lr=0.01, momentum=0.9, nesterov=True)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[20, 40, 55], gamma=0.1)

    # 断点续训：恢复被中断的 KD 训练，接着之前的 epoch 继续跑
    if RESUME:
        ckpt = torch.load(RESUME)
        student.load_state_dict(ckpt['net_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        # 将 LR scheduler 快进到正确的 epoch
        for _ in range(start_epoch - 1):
            exp_lr_scheduler.step()
        _print('KD training resumed from epoch {}'.format(start_epoch))

    student = student.cuda()
    teacher = teacher.cuda()
    ArcMargin = ArcMargin.cuda()
    if multi_gpus:
        student = DataParallel(student)
        teacher = DataParallel(teacher)
        ArcMargin = DataParallel(ArcMargin)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, TOTAL_EPOCH + 1):
        _print('Train Epoch: {}/{} ...'.format(epoch, TOTAL_EPOCH))
        student.train()

        train_total_loss = 0.0
        total = 0
        since = time.time()
        pbar = tqdm(trainloader, desc='  batch', ncols=110, leave=False)
        for data in pbar:
            img, label = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)
            optimizer_ft.zero_grad()

            student_feat = student(img)

            with torch.no_grad():
                teacher_feat = teacher(img)

            # CE loss via ArcFace
            output = ArcMargin(student_feat, label)
            ce_loss = criterion(output, label)

            # KD loss: cosine distance between normalized embeddings
            s_norm = F.normalize(student_feat, dim=1)
            t_norm = F.normalize(teacher_feat, dim=1)
            kd_loss = (1.0 - (s_norm * t_norm).sum(dim=1)).mean()

            total_loss = ce_loss + KD_LAMBDA * kd_loss
            total_loss.backward()
            optimizer_ft.step()

            train_total_loss += total_loss.item() * batch_size
            total += batch_size
            pbar.set_postfix({
                'ce': '{:.3f}'.format(ce_loss.item()),
                'kd': '{:.3f}'.format(kd_loss.item())
            })

        exp_lr_scheduler.step()
        train_total_loss = train_total_loss / total
        time_elapsed = time.time() - since
        _print('    total_loss: {:.4f} time: {:.0f}m {:.0f}s'.format(
            train_total_loss, time_elapsed // 60, time_elapsed % 60))

        # test model on lfw
        if epoch % TEST_FREQ == 0:
            student.eval()
            featureLs = None
            featureRs = None
            _print('Test Epoch: {} ...'.format(epoch))
            for data in testloader:
                for i in range(len(data)):
                    data[i] = data[i].cuda()
                res = [student(d).data.cpu().numpy() for d in data]
                featureL = np.concatenate((res[0], res[1]), 1)
                featureR = np.concatenate((res[2], res[3]), 1)
                if featureLs is None:
                    featureLs = featureL
                else:
                    featureLs = np.concatenate((featureLs, featureL), 0)
                if featureRs is None:
                    featureRs = featureR
                else:
                    featureRs = np.concatenate((featureRs, featureR), 0)

            result = {'fl': featureLs, 'fr': featureRs, 'fold': folds, 'flag': flags}
            scipy.io.savemat('./result/tmp_result.mat', result)
            accs = evaluation_10_fold('./result/tmp_result.mat')
            _print('    ave: {:.4f}'.format(np.mean(accs) * 100))

        # save model
        if epoch % SAVE_FREQ == 0:
            _print('Saving checkpoint: {}'.format(epoch))
            net_state_dict = student.module.state_dict() if multi_gpus else student.state_dict()
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save({
                'epoch': epoch,
                'net_state_dict': net_state_dict},
                os.path.join(save_dir, '%03d.ckpt' % epoch))

    print('finishing training')
