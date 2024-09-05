import os
import sys
sys.path.insert(0,'/scratch/smata/mask_diffusion/libraries/')
import wandb
import torch
import argparse
import warnings
import core.util as Util
import core.praser as Praser
import torch.multiprocessing as mp

from data import define_dataloader
from core.logger import VisualWriter, InfoLogger
from models import create_model, define_network, define_loss, define_metric

def main_worker(gpu, ngpus_per_node, opt):
    print('In main_worker():')

    # threads running on each GPU
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend = 'nccl', 
            init_method = opt['init_method'],
            world_size  = opt['world_size'], 
            rank        = opt['global_rank'],
            group_name  = 'mtorch'
        )

    # set seed and and cuDNN environment
    print('Setting seed and cuDNN environment...')
    torch.backends.cudnn.enabled = True
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    # set logger
    print('Setting logger and writer...')
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    # set networks and dataset
    print('Setting network and dataset...')
    phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
    n_phase_data = len(phase_loader.dataset)
    opt['n_phase_data'] = n_phase_data
    if (opt['global_rank'] == 0) & (opt['phase'] == 'train'): 
        n_val_data = len(val_loader.dataset)
        opt['n_val_data'] = n_val_data
        print(f'Using a phase_loader (that lives on all ranks) with n={n_phase_data} samples, and a val_loader (that lives only on rank 0) with n={n_val_data} samples')
    
    print('Defining network...')
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    # Set metrics, loss, optimizer and  schedulers
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses  = [define_loss(phase_logger, item_opt)   for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt          = opt,
        networks     = networks,
        phase_loader = phase_loader,
        val_loader   = val_loader,
        losses       = losses,
        metrics      = metrics,
        logger       = phase_logger,
        writer       = phase_writer
    )

    phase_logger.info('Begin model {}.'.format(opt['phase']))

    # Hardcode wandb logging
    if opt['global_rank'] == 0:
        print('Logging into wandb...')
        wandb.login()
        print('\twandb login successful!')

        run = wandb.init(
            # Set the project where this run will be logged
            project=opt['wandb_project'],
            name=opt['name'][6:16],
            # Track hyperparameters and run metadata
            config={
                'base_learning_rate': opt['model']['which_model']['args']['optimizers'][0]['lr'],
                'inner_channel':      opt['model']['which_networks'][0]['args']['unet']['inner_channel'],
                'beta_max':           opt['model']['which_networks'][0]['args']['beta_schedule']['train']['linear_end']
            },
        )

    try:
        if opt['phase'] == 'train':
            model.train()
        else:
            model.test()
    finally:
        if opt['train']['tensorboard']:
            phase_writer.close()
        
        
if __name__ == '__main__':
    print('Starting...')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',   '--config',  type = str, default = '/scratch/smata/mask_diffusion/config/settings.json', help = 'JSON file for configuration')
    parser.add_argument('-p',   '--phase',   type = str, choices = ['train','test'], help = 'Run train or test', default = 'train')
    parser.add_argument('-b',   '--batch',   type = int, default = None, help = 'Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type = str, default = None)
    parser.add_argument('-d',   '--debug', action = 'store_true')
    parser.add_argument('-P',   '--port', default = '21012', type = str)

    # parser configs
    print('Parsing arguments...')
    args = parser.parse_args()
    opt  = Praser.parse(args)
    
    # cuda devices
    print('Setting up cuda devices...')
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    # Use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:
        print('Running in distributed mode!')
        ngpus_per_node     = len(opt['gpu_ids']) # or torch.cuda.device_count()
        opt['world_size']  = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:'+ args.port 
        mp.spawn(main_worker, nprocs = ngpus_per_node, args = (ngpus_per_node, opt))
    else:
        print('NOT running in distributed mode')
        opt['world_size'] = 1 
        main_worker(0, 1, opt)