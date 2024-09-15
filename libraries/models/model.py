import tqdm
import copy
import torch
import wandb  # hardcode some wandb logging
import matplotlib.pyplot as plt
import torch.distributed as dist

from core.logger import LogTracker
from core.base_model import BaseModel
from torch.optim.lr_scheduler import CosineAnnealingLR

class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Palette(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        
        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        self.scheduler = CosineAnnealingLR(optimizer=self.optG,
                                           T_max=self.opt['train']['n_epoch'])  # TODO: Don't hardcode this scheduler
        self.schedulers.append(self.scheduler)
        self.resume_training() 

        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase, device=torch.device('cpu'))

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task
        
    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.mask = self.set_device(data.get('mask'))
        self.mask_image = data.get('mask_image')
        self.path = data['path']
        self.batch_size = len(data['path'])
    
    def get_current_visuals(self, phase='train'):
        dict = {
            'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
            'cond_image': (self.cond_image.detach()[:].float().cpu()+1)/2,
        }
        if self.task in ['inpainting','uncropping']:
            dict.update({
                'mask': self.mask.detach()[:].float().cpu(),
                'mask_image': (self.mask_image+1)/2,
            })
        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu()+1)/2
            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('Input_{}'.format(self.path[idx]))
            ret_result.append(self.cond_image[idx].detach().float().cpu())

            ret_path.append('GT_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('Process_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())
            
            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx-self.batch_size].detach().float().cpu())
        
        if self.task in ['inpainting','uncropping']:
            ret_path.extend(['Mask_{}'.format(name) for name in self.path])
            ret_result.extend(self.mask_image)

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        if self.opt['global_rank'] == 0:
            print("STORM AT TOP OF train_step in model.py; epoch is ", self.epoch)
        self.netG.train()
        self.train_metrics.reset()
        total_train_loss = torch.tensor(0.0).to(self.opt['global_rank'])  # Create a tensor to store the training loss on each GPU
        for train_data in tqdm.tqdm(self.phase_loader):
            self.set_input(train_data)
            self.optG.zero_grad()
            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask)
            total_train_loss += loss.item()
            loss.backward()
            self.optG.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals().items():
                    self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        # Log in wandb
        dist.all_reduce(total_train_loss, op=dist.ReduceOp.SUM)
        normalized_train_loss = total_train_loss / self.opt['n_phase_data']
        if self.opt['global_rank'] == 0:
            # Sum the training losses from all GPUs
            wandb_train_out_dict = {}
            current_lr = self.schedulers[0].get_last_lr()[0]  # TODO: Don't hardcode assumption of one scheduler
            wandb_train_out_dict['lr'] = current_lr
            wandb_train_out_dict['epoch'] = self.epoch
            wandb_train_out_dict['normalized_train_loss'] = normalized_train_loss
            wandb.log(wandb_train_out_dict)
        total_train_loss.zero_()

        for scheduler in self.schedulers:
            scheduler.step()

        if self.epoch % self.opt['train']['save_checkpoint_epoch'] == 0:
            self.logger.info('Saving the self at the end of epoch {:.0f}'.format(self.epoch))
            self.save_everything()

        return self.train_metrics.result()
    
    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            val_loss_like_train = 0
            val_loss_like_train_u = 0
            val_loss_like_train_v = 0
            val_loss_like_train_w = 0
            val_loss_like_train_T = 0
            val_loss_like_train_TKE = 0
            # Loop over entire val dataset and calculate basic loss
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                if self.opt['global_rank'] == 0:
                    val_loss_like_train += self.netG(self.gt_image, self.cond_image, mask=self.mask)
                    # if self.gt_image.shape[1] == 5:
                    #     val_loss_like_train_u += self.netG(self.gt_image[:,0,:,:], self.cond_image[:,0,:,:], mask=self.mask[0])
                    #     val_loss_like_train_v += self.netG(self.gt_image[:,1,:,:], self.cond_image[:,1,:,:], mask=self.mask[1])
                    #     val_loss_like_train_w += self.netG(self.gt_image[:,2,:,:], self.cond_image[:,2,:,:], mask=self.mask[2])
                    #     val_loss_like_train_T += self.netG(self.gt_image[:,3,:,:], self.cond_image[:,3,:,:], mask=self.mask[3])
                    #     val_loss_like_train_TKE += self.netG(self.gt_image[:,4,:,:], self.cond_image[:,4,:,:], mask=self.mask[4])

                    # self.iter += self.batch_size
                    # self.writer.set_iter(self.epoch, self.iter, phase='val')

            if self.opt['global_rank'] == 0:
                val_loss_like_train =  val_loss_like_train / self.opt['n_val_data']
                wandb.log({'val_loss_like_train': val_loss_like_train})
                # if self.gt_image.shape[1] == 5:
                #     val_loss_like_train_u =  val_loss_like_train_u / self.opt['n_val_data']
                #     val_loss_like_train_v =  val_loss_like_train_v / self.opt['n_val_data']
                #     val_loss_like_train_w =  val_loss_like_train_w / self.opt['n_val_data']
                #     val_loss_like_train_T =  val_loss_like_train_T / self.opt['n_val_data']
                #     val_loss_like_train_TKE =  val_loss_like_train_TKE / self.opt['n_val_data']
                #     wandb.log({'val_loss_like_train_u': val_loss_like_train_u,
                #                'val_loss_like_train_v': val_loss_like_train_v,
                #                'val_loss_like_train_w': val_loss_like_train_w,
                #                'val_loss_like_train_T': val_loss_like_train_T,
                #                'val_loss_like_train_TKE': val_loss_like_train_TKE})

            # Do the full restoration calculation, but just on the last batch
            assert self.cond_image.shape[0] == self.batch_size, "Adjust the number of validation files so that it's a multiple of the bachsize"
            if self.opt['distributed']:
                if self.task in ['inpainting','uncropping']:
                    self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                        y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                else:
                    self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
            else:
                if self.task in ['inpainting','uncropping']:
                    self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                        y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                else:
                    self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)

            if self.opt['global_rank'] == 0:       # wandb log
                wandb_val_out_dict = {}
                total_val_loss = self.metrics[0](self.gt_image, self.output)
                # total_val_loss_just_u = self.metrics[0](self.gt_image[:,0,:,:], self.output[:,0,:,:])
                normalized_val_loss = total_val_loss / self.batch_size
                wandb_val_out_dict['val_reconstruction2000_loss_one_batch'] = normalized_val_loss
                wandb.log(wandb_val_out_dict)

                u_xy_input = self.cond_image[:self.batch_size,0,:,:].cpu().float().numpy()
                u_xy_gt    = self.gt_image[:self.batch_size,0,:,:].cpu().float().numpy()
                u_xy_mask  = self.mask_image[:self.batch_size,0,:,:].cpu().float().numpy()
                u_xy_pred  = self.visuals[-self.batch_size:,0,:,:].cpu().float().numpy()
                fig_u_xy   = self.plot_cross_section_wandb(u_xy_input, u_xy_gt, u_xy_mask, u_xy_pred, self.batch_size)
                wandb.log({'u component': fig_u_xy})
                plt.close(fig_u_xy)

                if self.cond_image.shape[1] == 6:  # plot other variables if they are being reconstructed
                    v_xy_input = self.cond_image[:self.batch_size,1,:,:].cpu().float().numpy()
                    v_xy_gt    = self.gt_image[:self.batch_size,1,:,:].cpu().float().numpy()
                    v_xy_mask  = self.mask_image[:self.batch_size,1,:,:].cpu().float().numpy()
                    v_xy_pred  = self.visuals[-self.batch_size:,1,:,:].cpu().float().numpy()
                    fig_v_xy   = self.plot_cross_section_wandb(v_xy_input, v_xy_gt, v_xy_mask, v_xy_pred, self.batch_size)

                    hpdc_input = self.cond_image[:self.batch_size,2,:,:].cpu().float().numpy()
                    hpdc_mask  = self.mask_image[:self.batch_size,2,:,:].cpu().float().numpy()
                    hpdc_gt    = self.gt_image[:self.batch_size,2,:,:].cpu().float().numpy()
                    hpdc_pred  = self.visuals[-self.batch_size:,2,:,:].cpu().float().numpy()
                    fig_hpdc   = self.plot_cross_section_wandb(hpdc_input, hpdc_gt, hpdc_mask, hpdc_pred, self.batch_size)

                    hpds_input = self.cond_image[:self.batch_size,3,:,:].cpu().float().numpy()
                    hpds_gt    = self.gt_image[:self.batch_size,3,:,:].cpu().float().numpy()
                    hpds_mask  = self.mask_image[:self.batch_size,3,:,:].cpu().float().numpy()
                    hpds_pred  = self.visuals[-self.batch_size:,3,:,:].cpu().float().numpy()
                    fig_hpds   = self.plot_cross_section_wandb(hpds_input, hpds_gt, hpds_mask, hpds_pred, self.batch_size)

                    dpyc_input = self.cond_image[:self.batch_size,4,:,:].cpu().float().numpy()
                    dpyc_gt    = self.gt_image[:self.batch_size,4,:,:].cpu().float().numpy()
                    dpyc_mask  = self.mask_image[:self.batch_size,4,:,:].cpu().float().numpy()
                    dpyc_pred  = self.visuals[-self.batch_size:,4,:,:].cpu().float().numpy()
                    fig_dpyc   = self.plot_cross_section_wandb(dpyc_input, dpyc_gt, dpyc_mask, dpyc_pred, self.batch_size)

                    dpys_input = self.cond_image[:self.batch_size,5,:,:].cpu().float().numpy()
                    dpys_gt    = self.gt_image[:self.batch_size,5,:,:].cpu().float().numpy()
                    dpys_mask  = self.mask_image[:self.batch_size,5,:,:].cpu().float().numpy()
                    dpys_pred  = self.visuals[-self.batch_size:,5,:,:].cpu().float().numpy()
                    fig_dpys   = self.plot_cross_section_wandb(dpys_input, dpys_gt, dpys_mask, dpys_pred, self.batch_size)

                    wandb.log({'visual demo v_xy': fig_v_xy,
                               'visual demo hpdc': fig_hpdc,
                               'visual demo hpds': fig_hpds,
                               'visual demo dpyc': fig_dpyc,
                               'visual demo dpys': fig_dpys})
                    plt.close(fig_v_xy)
                    plt.close(fig_hpdc)
                    plt.close(fig_hpds)
                    plt.close(fig_dpyc)
                    plt.close(fig_dpys)
                
                elif self.cond_image.shape[1] == 2:
                    v_xy_input = self.cond_image[:self.batch_size,1,:,:].cpu().float().numpy()
                    v_xy_gt    = self.gt_image[:self.batch_size,1,:,:].cpu().float().numpy()
                    v_xy_mask  = self.mask_image[:self.batch_size,1,:,:].cpu().float().numpy()
                    v_xy_pred  = self.visuals[-self.batch_size:,1,:,:].cpu().float().numpy()
                    fig_v_xy   = self.plot_cross_section_wandb(v_xy_input, v_xy_gt, v_xy_mask, v_xy_pred, self.batch_size)

                    wandb.log({'visual demo v_xy': fig_v_xy})
                    plt.close(fig_v_xy)

                elif self.cond_image.shape[1] == 3:
                    v_xy_input = self.cond_image[:self.batch_size,1,:,:].cpu().float().numpy()
                    v_xy_gt    = self.gt_image[:self.batch_size,1,:,:].cpu().float().numpy()
                    v_xy_mask  = self.mask_image[:self.batch_size,1,:,:].cpu().float().numpy()
                    v_xy_pred  = self.visuals[-self.batch_size:,1,:,:].cpu().float().numpy()
                    fig_v_xy   = self.plot_cross_section_wandb(v_xy_input, v_xy_gt, v_xy_mask, v_xy_pred, self.batch_size)

                    hpd_input  = self.cond_image[:self.batch_size,2,:,:].cpu().float().numpy()
                    hpd_gt     = self.gt_image[:self.batch_size,2,:,:].cpu().float().numpy()
                    hpd_mask   = self.mask_image[:self.batch_size,2,:,:].cpu().float().numpy()
                    hpd_pred   = self.visuals[-self.batch_size:,2,:,:].cpu().float().numpy()
                    fig_hpd    = self.plot_cross_section_wandb(hpd_input, hpd_gt, hpd_mask, hpd_pred, self.batch_size)

                    wandb.log({'v component': fig_v_xy,
                               'time': fig_hpd})
                    plt.close(fig_v_xy)
                    plt.close(fig_time)

                elif self.cond_image.shape[1] == 4:
                    v_xy_input = self.cond_image[:self.batch_size,1,:,:].cpu().float().numpy()
                    v_xy_gt    = self.gt_image[:self.batch_size,1,:,:].cpu().float().numpy()
                    v_xy_mask  = self.mask_image[:self.batch_size,1,:,:].cpu().float().numpy()
                    v_xy_pred  = self.visuals[-self.batch_size:,1,:,:].cpu().float().numpy()
                    fig_v_xy   = self.plot_cross_section_wandb(v_xy_input, v_xy_gt, v_xy_mask, v_xy_pred, self.batch_size)

                    hpd_input  = self.cond_image[:self.batch_size,2,:,:].cpu().float().numpy()
                    hpd_gt     = self.gt_image[:self.batch_size,2,:,:].cpu().float().numpy()
                    hpd_mask   = self.mask_image[:self.batch_size,2,:,:].cpu().float().numpy()
                    hpd_pred   = self.visuals[-self.batch_size:,2,:,:].cpu().float().numpy()
                    fig_hpd    = self.plot_cross_section_wandb(hpd_input, hpd_gt, hpd_mask, hpd_pred, self.batch_size)

                    dpy_input  = self.cond_image[:self.batch_size,3,:,:].cpu().float().numpy()
                    dpy_gt     = self.gt_image[:self.batch_size,3,:,:].cpu().float().numpy()
                    dpy_mask   = self.mask_image[:self.batch_size,3,:,:].cpu().float().numpy()
                    dpy_pred   = self.visuals[-self.batch_size:,3,:,:].cpu().float().numpy()
                    fig_dpy    = self.plot_cross_section_wandb(dpy_input, dpy_gt, dpy_mask, dpy_pred, self.batch_size)

                    wandb.log({'v component': fig_v_xy,
                               'hour per day': fig_hpd,
                               'day per year': fig_dpy})
                    plt.close(fig_v_xy)
                    plt.close(fig_hpd)
                    plt.close(fig_dpy)

            for key, value in self.get_current_visuals(phase='val').items():
                self.writer.add_images(key, value)
            self.writer.save_images(self.save_current_results(), self.opt['datatype'])

        return self.val_metrics.result()

    def test(self):
        while self.epoch <= self.opt['test']['n_epoch']:
            self.epoch += 1
            self.netG.eval()
            self.test_metrics.reset()
            with torch.no_grad():
                for phase_data in tqdm.tqdm(self.phase_loader):
                    self.set_input(phase_data)
                    if self.opt['distributed']:
                        if self.task in ['inpainting','uncropping']:
                            self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                                y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                        else:
                            self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                    else:
                        if self.task in ['inpainting','uncropping']:
                            self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                                y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                        else:
                            self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
                            
                    self.iter += self.batch_size
                    self.writer.set_iter(self.epoch, self.iter, phase='test')
                    for met in self.metrics:
                        key = met.__name__
                        value = met(self.gt_image, self.output)
                        self.test_metrics.update(key, value)
                        self.writer.add_scalar(key, value)
                    for key, value in self.get_current_visuals(phase='test').items():
                        self.writer.add_images(key, value)
                    self.writer.save_images(self.save_current_results(), self.opt['datatype'])
            
            test_log = self.test_metrics.result()
            ''' save logged informations into log dict ''' 
            test_log.update({'epoch': self.epoch, 'iters': self.iter})

            ''' print logged informations to the screen and tensorboard ''' 
            for key, value in test_log.items():
                self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()

    def plot_cross_section_wandb(self, input_plane, data0_plane, data1_plane, data2_plane, batchsize):
        '''
        Plot a cross-section of data
        - data0_plane: a plane of data shaped [nbatch, nwhatever, nwhatever]
        '''
        batchsize = min(8, batchsize)

        if batchsize > 1:

            fig, ax = plt.subplots(5, batchsize, sharex = True, sharey = True, figsize = (batchsize*2.5, 8), dpi = 400)

            data3_plane      = data1_plane - data0_plane
            pltmin, pltmax   = data0_plane.min(), data0_plane.max()
            pltmin3, pltmax3 = data3_plane.min(), data3_plane.max()

            for i, axs in enumerate(ax[0,:]):
                axs.set_title(f'Input, Sample {i}')
                im0 = axs.imshow(input_plane[i,:,:].T,
                        vmin   = pltmin,
                        vmax   = pltmax,
                        origin = 'lower')
                fig.colorbar(im0, ax = axs) 
            for i, axs in enumerate(ax[1,:]):
                axs.set_title(f'GT, Sample {i}')
                im1 = axs.imshow(data0_plane[i,:,:].T,
                        vmin   = pltmin,
                        vmax   = pltmax,
                        origin = 'lower')
                fig.colorbar(im1, ax = axs) 
            for i, axs in enumerate(ax[2,:]):
                axs.set_title(f'Mask, Sample {i}')
                im2 = axs.imshow(data1_plane[i,:,:].T,
                        vmin   = 0,
                        vmax   = 1,
                        origin = 'lower')
                fig.colorbar(im2, ax = axs) 
            for i, axs in enumerate(ax[3,:]):
                axs.set_title(f'Pred, Sample {i}')
                im3 = axs.imshow(data2_plane[i,:,:].T,
                        vmin   = pltmin,
                        vmax   = pltmax,
                        origin = 'lower')
                fig.colorbar(im3, ax = axs) 
            for i, axs in enumerate(ax[4,:]):
                axs.set_title(f'Diff, Sample {i}')
                im4 = axs.imshow(data3_plane[i,:,:].T,
                        vmin   = -1,
                        vmax   = 1,
                        cmap   = 'RdBu_r',
                        origin = 'lower')  
                fig.colorbar(im4, ax = axs) 

        else:

            fig, ax = plt.subplots(1, 5, sharey = True, figsize = (batchsize*2, 8), dpi = 600)

            data3_plane      = data1_plane - data0_plane
            pltmin, pltmax   = data0_plane.min(), data0_plane.max()
            pltmin3, pltmax3 = data3_plane.min(), data3_plane.max()

            ax[0].set_title(f'Input')
            im0 = ax[0].imshow(input_plane[:,:].T,
                    vmin   = pltmin,
                    vmax   = pltmax,
                    origin = 'lower')
            fig.colorbar(im0, ax = ax[0])
            ax[1].set_title(f'GT')
            im1 = ax[1].imshow(data0_plane[:,:].T,
                    vmin   = pltmin,
                    vmax   = pltmax,
                    origin = 'lower')
            fig.colorbar(im1, ax = ax[1])
            ax[2].set_title(f'Mask')
            im2 = ax[2].imshow(data1_plane[:,:].T,
                    vmin   = 0,
                    vmax   = 1,
                    origin = 'lower')
            fig.colorbar(im2, ax = ax[2])
            ax[3].set_title(f'Pred')
            im3 = ax[3].imshow(data2_plane[:,:].T,
                    vmin   = pltmin,
                    vmax   = pltmax,
                    origin = 'lower')
            fig.colorbar(im3, ax = ax[3])
            ax[4].set_title(f'Diff')
            im4 = ax[4].imshow(data3_plane[:,:].T,
                    vmin   = -1,
                    vmax   = 1,
                    cmap   = 'RdBu_r',
                    origin = 'lower')   
            fig.colorbar(im4, ax = ax[4])    

        return fig
