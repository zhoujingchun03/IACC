import os
import torch
import random
import numpy as np
import yaml
from ruamel.yaml import YAML
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from ptflops import get_model_complexity_info


class MetricRecorder:
    def __init__(self):
        self.avg = .0
        self.count = 0
        self.value = .0
        self.total = .0

    def reset(self):
        self.avg = .0
        self.count = 0
        self.value = .0
        self.total = .0

    def update(self, value):
        value = round(value, 4)
        self.value = value
        self.total += value
        self.count += 1
        self.avg = round(self.total / self.count, 4)


def save_pics(hparams, source_img, target_img, output_img):
    base_path = os.path.join(hparams['train']['save_dir'], hparams['train']['model_name'],
                             hparams['train']['task_name'], 'pic')
    src = make_grid(source_img)
    tar = make_grid(target_img)
    out = make_grid(output_img)
    save_image(src, os.path.join(base_path, 'source.png'))
    save_image(tar, os.path.join(base_path, 'target.png'))
    save_image(out, os.path.join(base_path, 'output.png'))


def set_all_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_all(epoch, model, optimizer, scheduler, scaler,
             hparams, best_metric, name):
    torch.save({
        'end_epoch': epoch,
        'best_ssim': best_metric['ssim']['value'],
        'ssim_epoch': best_metric['ssim']['epoch'],
        'best_psnr': best_metric['psnr']['value'],
        'psnr_epoch': best_metric['psnr']['epoch'],
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict()
    }, os.path.join(
        hparams['train']['save_dir'],
        hparams['train']['model_name'],
        hparams['train']['task_name'],
        'ckpt', name + '.pth'
        )
    )


def load_all(hparams, name, model, optimizer, scheduler, scaler, best_metric):
    ckpt_path = os.path.join(hparams['train']['save_dir'],
                             hparams['train']['model_name'],
                             hparams['train']['task_name'],
                             'ckpt', name + '.pth')
    ckpt_info = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt_info['model'])
    optimizer.load_state_dict(ckpt_info['optimizer'])
    scheduler.load_state_dict(ckpt_info['scheduler'])
    scaler.load_state_dict(ckpt_info['scaler'])
    best_metric['ssim']['value'] = ckpt_info['best_ssim']
    best_metric['ssim']['epoch'] = ckpt_info['ssim_epoch']
    best_metric['psnr']['value'] = ckpt_info['best_psnr']
    best_metric['psnr']['epoch'] = ckpt_info['psnr_epoch']
    return ckpt_info['end_epoch']


def print_epoch_result(train_result, valid_result, epoch):
    print('')
    print('Epoch:  {}'.format(epoch))
    for key, value in train_result.items():
        print(key, ':  ', value)
    if valid_result is not None:
        for key, value in valid_result.items():
            print(key, ':  ', value)


def parse_yaml(path: str):
    yaml = YAML(typ='safe')
    with open(path, 'r') as f:
        args = yaml.load(f)
    return args


def save_dict_as_yaml(hparams, save_path):
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        f.write(yaml.dump(hparams, allow_unicode=True))


class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_multi_scaler(self, scaler_dict, epoch):
        for key, value in scaler_dict.items():
            self.writer.add_scalar(key, value, epoch)


def make_all_dirs(hparams):
    base_dir = os.path.join(hparams['train']['save_dir'], hparams['train']['model_name'],
                            hparams['train']['task_name'])
    if not os.path.exists(os.path.join(base_dir, 'tensorboard')):
        os.makedirs(os.path.join(base_dir, 'tensorboard'))
    if not os.path.exists(os.path.join(base_dir, 'ckpt')):
        os.makedirs(os.path.join(base_dir, 'ckpt'))
    if not os.path.exists(os.path.join(base_dir, 'pic')):
        os.makedirs(os.path.join(base_dir, 'pic'))


def print_params_and_macs(model):
    macs, params = get_model_complexity_info(model, (3, 256, 256), verbose=False, print_per_layer_stat=False)
    print('MACS: ' + str(macs))
    print('Params: ' + str(params))


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}