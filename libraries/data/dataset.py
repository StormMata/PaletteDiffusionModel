import os
import torch
import numpy as np
import torch.utils.data as data

from PIL import Image
from torchvision import transforms
from .util.mask import (bottom_mask, bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
ARR_EXTENSIONS = ['.npy']
TENSOR_EXTENSIONS = ['.pt']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_array_file(filename):
    return any(filename.endswith(extension) for extension in ARR_EXTENSIONS)

def is_tensor_file(filename):
    return any(filename.endswith(extension) for extension in TENSOR_EXTENSIONS)

def make_dataset(dir, filetype='image'):
    if os.path.isfile(dir):
        samples = [i for i in np.genfromtxt(dir, dtype=str, encoding='utf-8')]
    else:
        samples = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if filetype == 'image':
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        samples.append(path)
                elif filetype == 'array':
                    if is_array_file(fname):
                        path = os.path.join(root, fname)
                        samples.append(path)
                elif filetype == 'tensor':
                    if is_tensor_file(fname):
                        path = os.path.join(root, fname)
                        samples.append(path)
                else:
                    raise ValueError(f'Filetype of "{filetype}" not recognized')           

    return samples

def pil_loader(path):
    return Image.open(path).convert('RGB')

def numpy_loader(path):
    return np.load(path)

def pytorch_loader(path):
    return torch.load(path)

def numpy_transforms(arr, data_bounds):
    '''
    Inputs: 
      - arr: Numpy array, where data comes from natural values
      - data_bounds: Tuple containing values for rescaling (umin, umax, vmin, vmax[, wmin, wmax])
    Output: Torch tensor rescaled to [-1,1]
    '''

    # Rescale to [-1,1]
    if len(arr.shape) == 3:  # For planar data
        # First drop q but retain u&v for the QG data
        arr = arr[1:,:,:]  # Drop q, retain u+v

        # Then rescale
        umin, umax, vmin, vmax = data_bounds
        arr[0,:,:] = 2*(arr[0,:,:] - umin)/(umax - umin) - 1
        arr[1,:,:] = 2*(arr[1,:,:] - vmin)/(vmax - vmin) - 1

    elif len(arr.shape) == 4:  # For volume data
        raise NotImplementedError

    else:
        raise ValueError("Expected tuple containing (umin, umax, vmin, vmax[, wmin, wmax])!")

    # numpy array -> torch tensor with correct dtype
    arr = torch.from_numpy(arr).float()

    return arr

def tensor_transforms(tensors, data_bounds):
    '''
    Same as numpy_transforms, except for when you're loading
      PyTorch Tensors instead of numpy arrays
    '''

    x = tensors['x']
    y = tensors['y']

    if x.shape[0] == 1:
        # Rescale to [-1,1]
        umin, umax = data_bounds

        # Check that data is within expected bounds
        assert (x[0,:,:].min() >= umin) & (x[0,:,:].max() <= umax), f"Unexpected u values! min/max = {x[0,:,:].min()}/{x[0,:,:].max()}"

        x[0,:,:] = 2*(x[0,:,:] - umin)/(umax - umin) - 1

        y[0,:,:] = 2*(y[0,:,:] - umin)/(umax - umin) - 1

    elif x.shape[0] == 5:
        assert x.shape[0] == 5, f"tensor_transforms assumes 5 channel data! Shape of x is {x.shape}"

        # Rescale to [-1,1]
        umin, umax, vmin, vmax, wmin, wmax, Tmin, Tmax, TKEmin, TKEmax = data_bounds

        print("ALEX DEBUGGING")
        print("min/max before scaling")
        print("u:", x[0].min(), x[0].max())
        print("v:", x[1].min(), x[1].max())
        print("w:", x[2].min(), x[2].max())
        print("T:", x[3].min(), x[3].max())
        print("TKE:", x[4].min(), x[4].max())

        # Check that data is within expected bounds
        assert (x[0,:,:].min() >= umin) & (x[0,:,:].max() <= umax), f"Unexpected u values! min/max = {x[0,:,:].min()}/{x[0,:,:].max()}"
        assert (x[1,:,:].min() >= vmin) & (x[1,:,:].max() <= vmax), f"Unexpected v values! min/max = {x[1,:,:].min()}/{x[1,:,:].max()}"
        assert (x[2,:,:].min() >= wmin) & (x[2,:,:].max() <= wmax), f"Unexpected w values! min/max = {x[2,:,:].min()}/{x[2,:,:].max()}"
        assert (x[3,:,:].min() >= Tmin) & (x[3,:,:].max() <= Tmax), f"Unexpected T values! min/max = {x[3,:,:].min()}/{x[3,:,:].max()}"
        assert (x[4,:,:].min() >= TKEmin) & (x[4,:,:].max() <= TKEmax), f"Unexpected TKE values! min/max = {x[4,:,:].min()}/{x[4,:,:].max()}"

        x[0,:,:] = 2*(x[0,:,:] - umin)/(umax - umin) - 1
        x[1,:,:] = 2*(x[1,:,:] - vmin)/(vmax - vmin) - 1
        x[2,:,:] = 2*(x[2,:,:] - wmin)/(wmax - wmin) - 1
        x[3,:,:] = 2*(x[3,:,:] - Tmin)/(Tmax - Tmin) - 1
        x[4,:,:] = 2*(x[4,:,:] - TKEmin)/(TKEmax - TKEmin) - 1

        y[0,:,:] = 2*(y[0,:,:] - umin)/(umax - umin) - 1
        y[1,:,:] = 2*(y[1,:,:] - vmin)/(vmax - vmin) - 1
        y[2,:,:] = 2*(y[2,:,:] - wmin)/(wmax - wmin) - 1
        y[3,:,:] = 2*(y[3,:,:] - Tmin)/(Tmax - Tmin) - 1
        y[4,:,:] = 2*(y[4,:,:] - TKEmin)/(TKEmax - TKEmin) - 1

        print("min/max after scaling")
        print("u:", x[0].min(), x[0].max())
        print("v:", x[1].min(), x[1].max())
        print("w:", x[2].min(), x[2].max())
        print("T:", x[3].min(), x[3].max())
        print("TKE:", x[4].min(), x[4].max())

    elif x.shape[0] == 6:
        assert x.shape[0] == 6, f"tensor_transforms assumes 6 channel data! Shape of x is {x.shape}"

        # Rescale to [-1,1]
        umin, umax, vmin, vmax, hpdcmin, hpdcmax, hpdsmin, hpdsmax, dpycmin, dpycmax, dpysmin, dpysmax = data_bounds

        print("STORM DEBUGGING")
        print("min/max before scaling")
        print("u:",    x[0].min(), x[0].max())
        print("v:",    x[1].min(), x[1].max())
        #print("hpdc:", x[2].min(), x[2].max())
        #print("hpds:", x[3].min(), x[3].max())
        #print("dpyc:", x[4].min(), x[4].max())
        #print("dpys:", x[5].min(), x[5].max())

        # Check that data is within expected bounds
        assert (x[0,:,:].min() >= umin)    & (x[0,:,:].max() <= umax),    f"Unexpected u values! min/max    = {x[0,:,:].min()}/{x[0,:,:].max()}"
        assert (x[1,:,:].min() >= vmin)    & (x[1,:,:].max() <= vmax),    f"Unexpected v values! min/max    = {x[1,:,:].min()}/{x[1,:,:].max()}"
        #assert (x[2,:,:].min() >= hpdcmin) & (x[2,:,:].max() <= hpdcmax), f"Unexpected hpdc values! min/max = {x[2,:,:].min()}/{x[2,:,:].max()}"
        #assert (x[3,:,:].min() >= hpdsmin) & (x[3,:,:].max() <= hpdsmax), f"Unexpected hpds values! min/max = {x[3,:,:].min()}/{x[3,:,:].max()}"
        #assert (x[4,:,:].min() >= dpycmin) & (x[4,:,:].max() <= dpycmax), f"Unexpected dpyc values! min/max = {x[4,:,:].min()}/{x[4,:,:].max()}"
        #assert (x[5,:,:].min() >= dpysmin) & (x[5,:,:].max() <= dpysmax), f"Unexpected dpys values! min/max = {x[5,:,:].min()}/{x[5,:,:].max()}"

        x[0,:,:] = 2*(x[0,:,:] - umin)/(umax - umin) - 1
        x[1,:,:] = 2*(x[1,:,:] - vmin)/(vmax - vmin) - 1
        #x[2,:,:] = 2*(x[2,:,:] - hpdcmin)/(hpdcmax - hpdcmin) - 1
        #x[3,:,:] = 2*(x[3,:,:] - hpdsmin)/(hpdsmax - hpdsmin) - 1
        #x[4,:,:] = 2*(x[4,:,:] - dpycmin)/(dpycmax - dpycmin) - 1
        #x[5,:,:] = 2*(x[5,:,:] - dpysmin)/(dpysmax - dpysmin) - 1

        y[0,:,:] = 2*(y[0,:,:] - umin)/(umax - umin) - 1
        y[1,:,:] = 2*(y[1,:,:] - vmin)/(vmax - vmin) - 1
        #y[2,:,:] = 2*(y[2,:,:] - hpdcmin)/(hpdcmax - hpdcmin) - 1
        #y[3,:,:] = 2*(y[3,:,:] - hpdsmin)/(hpdsmax - hpdsmin) - 1
        #y[4,:,:] = 2*(y[4,:,:] - dpycmin)/(dpycmax - dpycmin) - 1
        #y[5,:,:] = 2*(y[5,:,:] - dpysmin)/(dpysmax - dpysmin) - 1

        print("min/max after scaling")
        print("u:", x[0].min(), x[0].max())
        print("v:", x[1].min(), x[1].max())
        #print("hpdc:", x[2].min(), x[2].max())
        #print("hpds:", x[3].min(), x[3].max())
        #print("dpyc:", x[4].min(), x[4].max())
        #print("dpys:", x[5].min(), x[5].max())

    else:
        raise ValueError(f"Expected 1 or 5 channel data, but got {x.shape[0]} channels")

    return x, y

class InpaintQGDataset(data.Dataset):
    def __init__(self, data_root, data_bounds, mask_config={}, data_len=-1, image_size=[64, 64], loader=numpy_loader):
        imgs = make_dataset(data_root, filetype='array')  # Refer to samples as "imgs" for simplicity's sake
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = numpy_transforms
        self.loader = loader
        self.data_bounds = data_bounds
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        if 'mask_dir' in mask_config.keys():
            self.mask_dir = self.mask_config['mask_dir']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path), self.data_bounds)
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            mask = get_custom_mask(self.image_size, self.mask_dir)
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)
            
class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)

class InpaintProfiles(data.Dataset):
    def __init__(self, data_root, data_len, mask_config, data_bounds, image_size, loader=pytorch_loader):
        imgs = make_dataset(data_root, filetype='tensor')
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs         = tensor_transforms
        self.loader      = loader
        self.data_bounds = data_bounds
        self.image_size  = image_size
        self.mask_config = mask_config
        self.mask_mode   = self.mask_config['mask_mode']

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        x, y = self.tfs(self.loader(path), self.data_bounds)
        mask = self.get_mask()
        # cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        # mask_img = img*(1. - mask) + mask

        ret['gt_image'] = y
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)

class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)

class LidarImg2ImgDataset(data.Dataset):
    def __init__(self, data_root, data_len, mask_config, data_bounds, image_size, loader=pytorch_loader):
        imgs = make_dataset(data_root, filetype='tensor')  # Refer to samples as "imgs" for simplicity's sake
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs         = tensor_transforms
        self.loader      = loader
        self.data_bounds = data_bounds
        self.image_size  = image_size
        self.mask_config = mask_config
        self.mask_mode   = self.mask_config['mask_mode']
    

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        x, y = self.tfs(self.loader(path), self.data_bounds)
        mask = self.get_mask()
        # raise NotImplementedError(f'Mask shape is {mask.shape}, x shape is {x.shape}, and y shape is {y.shape}.')
        # cond_image = y*(1. - mask) + mask*torch.randn_like(y)
        # mask_img = y * (1. - mask) + mask

        # Expand mask to match the shape of y
        mask_expanded = mask.expand(y.shape)  # mask_expanded will have shape [6, 96, 200]
        # mask_expanded = mask


        # Now you can safely perform operations between y and mask_expanded
        mask_img = y * (1. - mask_expanded) + mask_expanded
        cond_image = y*(1. - mask_expanded) + mask_expanded*torch.randn_like(y)

        # mask_img = x * mask

        ret['gt_image']   = y
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask_expanded
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        # raise NotImplementedError('Terminated here.')
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bottom':
            mask = bottom_mask(self.image_size)
        return torch.from_numpy(mask).permute(2,0,1)
    

    # def get_mask(self):
    #     if self.mask_mode == 'bbox':
    #         mask = bbox2mask(self.image_size, random_bbox())
    #     elif self.mask_mode == 'center':
    #         h, w = self.image_size
    #         mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
    #     elif self.mask_mode == 'irregular':
    #         mask = get_irregular_mask(self.image_size)
    #     elif self.mask_mode == 'free_form':
    #         mask = brush_stroke_mask(self.image_size)
    #     elif self.mask_mode == 'hybrid':
    #         regular_mask = bbox2mask(self.image_size, random_bbox())
    #         irregular_mask = brush_stroke_mask(self.image_size, )
    #         mask = regular_mask | irregular_mask
    #     elif self.mask_mode == 'file':
    #         pass
    #     else:
    #         raise NotImplementedError(
    #             f'Mask mode {self.mask_mode} has not been implemented.')
    #     return torch.from_numpy(mask).permute(2,0,1)