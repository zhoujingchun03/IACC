import os
from engine.dehaze import train
from data.uieb import UIEBTrain, UIEBValid
from torch.utils.data import DataLoader
from timm.optim import AdamW
from timm.scheduler import CosineLRScheduler
from model.base import CLCC
from utils.common_utils import parse_yaml, Logger, print_params_and_macs, save_dict_as_yaml
from torch.cuda.amp import GradScaler


def configuration_dataloader(hparams, stage_index):
    train_dataset = UIEBTrain(
        folder=hparams['data']['train_path'],
        size=hparams['data']['train_img_size'][stage_index]
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=hparams['data']['train_batch_size'][stage_index],
        shuffle=True,
        num_workers=hparams['data']['num_workers'],
        pin_memory=hparams['data']['pin_memory']
    )
    valid_dataset = UIEBValid(
        folder=hparams['data']['valid_path'],
        size=256
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        num_workers=hparams['data']['num_workers'],
        pin_memory=hparams['data']['pin_memory']
    )
    return train_loader, valid_loader


def configuration_dataloader2(hparams, stage_index):
    train_dataset = UIEBValid(
        folder=hparams['data']['train_path'],
        size=256
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=hparams['data']['train_batch_size'][stage_index],
        shuffle=True,
        num_workers=hparams['data']['num_workers'],
        pin_memory=hparams['data']['pin_memory']
    )
    valid_dataset = UIEBValid(
        folder=hparams['data']['valid_path'],
        size=256
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        num_workers=hparams['data']['num_workers'],
        pin_memory=hparams['data']['pin_memory']
    )
    return train_loader, valid_loader


def configuration_optimizer(model, hparams):
    total_epochs = sum(hparams['train']['stage_epochs'])
    optimizer = AdamW(
        params=model.parameters(),
        lr=hparams['optim']['lr_init'],
        weight_decay=hparams['optim']['weight_decay']
    )
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=(sum(hparams['train']['stage_epochs']) // len(hparams['train']['stage_epochs'])
                   if hparams['optim']['use_cycle_limit'] else sum(hparams['train']['stage_epochs'])),
        lr_min=hparams['optim']['lr_min'],
        cycle_limit=len(hparams['train']['stage_epochs']) if hparams['optim']['use_cycle_limit'] else 1,
        cycle_decay=hparams['optim']['cycle_decay'],
        warmup_t=hparams['optim']['warmup_epochs'],
        warmup_lr_init=hparams['optim']['lr_min']
    )
    return optimizer, scheduler


if __name__ == '__main__':
    args = parse_yaml(r'./config.yaml')
    base_path = os.path.join(args['train']['save_dir'],
                             args['train']['model_name'],
                             args['train']['task_name'])

    model = CLCC(64, 3, 3).cuda()
    scaler = GradScaler()
    logger = Logger(os.path.join(base_path, 'tensorboard'))
    optimizer, scheduler = configuration_optimizer(model, args)
    save_dict_as_yaml(args, base_path)
    print_params_and_macs(model)

    for i in range(len(args['train']['stage_epochs'])):
        print('\033[92m\nStart Stage {}'.format(i + 1))
        if i != 0:
            args['train']['resume'] = True
        train_loader, valid_loader = configuration_dataloader2(args, i)
        train(args, model, optimizer, scaler, scheduler, logger, train_loader, valid_loader, i)
        print('\033[92mEndStage {}\n'.format(i + 1))
