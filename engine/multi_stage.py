# type hint
from typing import Dict
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.nn import L1Loss, KLDivLoss

from loss.perceptual import PerceptualLoss
from loss.ssim import SSIMLoss
from utils.common_utils import MetricRecorder
from kornia.metrics import ssim, psnr
from utils.common_utils import (save_pics, set_all_seed, save_all,
                                load_all, print_epoch_result, make_all_dirs)
import os
from torchvision.utils import save_image, make_grid


def train_one_epoch(
        hparams: Dict,
        model: Module,
        scaler: GradScaler,
        optimizer: Optimizer,
        scheduler,
        train_loader: DataLoader
):
    model.train()
    loss_recoder = MetricRecorder()
    l1_criterion = L1Loss().to(hparams['train']['device'])
    ssim_criterion = SSIMLoss().to(hparams['train']['device'])
    prec_criterion = PerceptualLoss().to(hparams['train']['device'])
    for batch in tqdm(train_loader, ncols=120):
        source_img, target_img = batch
        source_img = source_img.to(hparams['train']['device'])
        target_img = target_img.to(hparams['train']['device'])
        with autocast(hparams['train']['use_amp']):
            output_img = model(source_img)
            loss = 0.3 * l1_criterion(output_img[0], target_img) + 0.6 * l1_criterion(output_img[1],
                                                                                      target_img) + l1_criterion(
                output_img[2], target_img) + ssim_criterion(output_img[2], target_img) + 0.02 * prec_criterion(
                output_img[2],
                target_img)
        optimizer.zero_grad()
        if hparams['train']['use_amp']:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_recoder.update(loss.item())
    return {'train_loss': loss_recoder.avg, 'lr': optimizer.param_groups[0]['lr']}


def valid(
        hparams: Dict,
        model: Module,
        valid_loader: DataLoader
):
    model.eval()
    loss_recorder = MetricRecorder()
    ssim_recorder = MetricRecorder()
    psnr_recorder = MetricRecorder()
    l1_criterion = L1Loss().to(hparams['train']['device'])
    ssim_criterion = SSIMLoss().to(hparams['train']['device'])
    prec_criterion = PerceptualLoss().to(hparams['train']['device'])
    for i, batch in enumerate(valid_loader):
        source_img, target_img = batch
        source_img = source_img.to(hparams['train']['device'])
        target_img = target_img.to(hparams['train']['device'])
        with torch.no_grad():
            output_img = model(source_img)
        for j in range(len(output_img)):
            output_img[j] = output_img[j].clamp(0, 1)
        loss = 0.3 * l1_criterion(output_img[0], target_img) + 0.6 * l1_criterion(output_img[1],
                                                                                  target_img) + l1_criterion(
            output_img[2], target_img) + ssim_criterion(output_img[2], target_img) + 0.02 * prec_criterion(
            output_img[2], target_img)
        loss_recorder.update(loss.item())
        ssim_recorder.update(ssim(output_img[2], target_img, 5).mean().item())
        psnr_recorder.update(psnr(output_img[2], target_img, 1).item())
        if i % 5 == 0:
            base_path = os.path.join(hparams['train']['save_dir'], hparams['train']['model_name'],
                                     hparams['train']['task_name'], 'pic')
            src = make_grid(source_img)
            tar = make_grid(target_img)
            group1 = make_grid(output_img[0])
            group2 = make_grid(output_img[1])
            group3 = make_grid(output_img[2])
            save_image(src, os.path.join(base_path, 'source.png'))
            save_image(tar, os.path.join(base_path, 'target.png'))
            save_image(group1, os.path.join(base_path, 'group1.png'))
            save_image(group2, os.path.join(base_path, 'group2.png'))
            save_image(group3, os.path.join(base_path, 'group3.png'))
    return {'valid_loss': loss_recorder.avg, 'ssim': ssim_recorder.avg,
            'psnr': psnr_recorder.avg}


def train(
        hparams: Dict,
        model: Module,
        optimizer: Optimizer,
        scaler: GradScaler,
        scheduler,
        logger,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        stage_index: int
):
    set_all_seed(hparams['train']['seed'])
    make_all_dirs(hparams)
    best_metric = {'ssim': {'value': .0, 'epoch': 0},
                   'psnr': {'value': .0, 'epoch': 0}}
    if hparams['train']['resume']:
        print('==========>Start Resume<==========')
        start_epoch = load_all(hparams, hparams['train']['ckpt_name'], model,
                               optimizer, scheduler, scaler, best_metric) + 1
    else:
        print('==========>Start Training<==========')
        start_epoch = 1
    print('Since from {}th Epoch'.format(start_epoch))
    max_epochs = sum(hparams['train']['stage_epochs'][: stage_index + 1])
    for current_epoch in range(start_epoch, max_epochs + 1):
        train_return = train_one_epoch(hparams, model, scaler, optimizer, scheduler, train_loader)
        logger.log_multi_scaler(train_return, current_epoch)
        scheduler.step(current_epoch)
        # valid
        valid_result = None
        if current_epoch % hparams['train']['valid_frequency'] == 0:
            valid_result = valid(hparams, model, valid_loader)
            logger.log_multi_scaler(valid_result, current_epoch)
            if valid_result['ssim'] > best_metric['ssim']['value']:
                best_metric['ssim']['value'] = valid_result['ssim']
                best_metric['ssim']['epoch'] = current_epoch
                save_all(current_epoch, model, optimizer, scheduler, scaler,
                         hparams, best_metric, 'best_ssim')
            if valid_result['psnr'] > best_metric['psnr']['value']:
                best_metric['psnr']['value'] = valid_result['psnr']
                best_metric['psnr']['epoch'] = current_epoch
                save_all(current_epoch, model, optimizer, scheduler, scaler,
                         hparams, best_metric, 'best_psnr')
        save_all(current_epoch, model, optimizer, scheduler, scaler,
                 hparams, best_metric, 'last')
        if max_epochs - 5 <= current_epoch <= max_epochs - 1:
            save_all(current_epoch, model, optimizer, scheduler, scaler,
                     hparams, best_metric, 'epoch{}_psnr{}_ssim{}'.format(current_epoch,
                                                                          valid_result['psnr'],
                                                                          valid_result['ssim']))
        print_epoch_result(train_return, valid_result, current_epoch)
        print('best ssim: ', best_metric['ssim']['value'], '  best epoch: ', best_metric['ssim']['epoch'])
        print('best psnr: ', best_metric['psnr']['value'], '  best epoch: ', best_metric['psnr']['epoch'])
