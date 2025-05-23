import time

import cv2
import numpy as np
import torch
# import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from yacs.config import CfgNode as CN


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def upper_config(dict_cfg):
    if not isinstance(dict_cfg, dict):
        return dict_cfg
    return {k.upper(): upper_config(v) for k, v in dict_cfg.items()}


class DataIOWrapper(nn.Module):
    """
    Pre-propcess data from different sources
    """

    def __init__(self, model, config, ckpt=None):
        super().__init__()

        self.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
        torch.set_grad_enabled(False)
        self.model = model
        self.config = config
        self.img0_size = config['img0_resize']
        self.img1_size = config['img1_resize']
        self.df = config['df']
        self.padding = config['padding']
        self.coarse_scale = config['coarse_scale']

        self.model = self.model.eval().to(self.device)

    def preprocess_image(self, img, device, resize=None, df=None, padding=None, cam_K=None, dist=None, gray_scale=True):
        # xoftr takes grayscale input images
        if gray_scale and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = img.shape[:2]
        new_K = None
        img_undistorted = None
        if cam_K is not None and dist is not None:
            new_K, roi = cv2.getOptimalNewCameraMatrix(cam_K, dist, (w, h), 0, (w, h))
            img = cv2.undistort(img, cam_K, dist, None, new_K)
            img_undistorted = img.copy()

        if resize is not None:
            scale = resize / max(h, w)
            w_new, h_new = int(round(w * scale)), int(round(h * scale))
        else:
            w_new, h_new = w, h

        if df is not None:
            w_new, h_new = map(lambda x: int(x // df * df), [w_new, h_new])

        img = cv2.resize(img, (w_new, h_new))
        scale = np.array([w / w_new, h / h_new], dtype=float)

        if padding:  # padding
            pad_to = max(h_new, w_new)
            img, mask = self.pad_bottom_right(img, pad_to, ret_mask=True)
            mask = torch.from_numpy(mask).to(device)
        else:
            mask = None
        # img = transforms.functional.to_tensor(img).unsqueeze(0).to(device)
        if len(img.shape) == 2:  # grayscale image
            img = torch.from_numpy(img)[None][None].to(self.device).float() / 255.0
        else:  # Color image
            img = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
        return img, scale, mask, new_K, img_undistorted

    def from_cv_imgs(self, img0, img1, K0=None, K1=None, dist0=None, dist1=None):
        # print('self.padding', self.padding)
        img0_tensor, scale0, mask0, new_K0, img0_undistorted = self.preprocess_image(
            img0, self.device, resize=self.img0_size, df=self.df, padding=self.padding, cam_K=K0, dist=dist0)
        img1_tensor, scale1, mask1, new_K1, img1_undistorted = self.preprocess_image(
            img1, self.device, resize=self.img1_size, df=self.df, padding=self.padding, cam_K=K1, dist=dist1)
        mkpts0, mkpts1, mconf, match_time = self.match_images(img0_tensor, img1_tensor, mask0, mask1)

        mkpts0 = mkpts0 * scale0
        mkpts1 = mkpts1 * scale1

        matches = np.concatenate([mkpts0, mkpts1], axis=1)
        data = {'matches': matches,
                'mkpts0': mkpts0,
                'mkpts1': mkpts1,
                'mconf': mconf,
                'img0': img0,
                'img1': img1,
                'match_time': match_time
                }
        if K0 is not None and dist0 is not None:
            data.update({'new_K0': new_K0, 'img0_undistorted': img0_undistorted})
        if K1 is not None and dist1 is not None:
            data.update({'new_K1': new_K1, 'img1_undistorted': img1_undistorted})
        return data

    def from_paths(self, img0_pth, img1_pth, K0=None, K1=None, dist0=None, dist1=None, read_color=False):

        imread_flag = cv2.IMREAD_COLOR if read_color else cv2.IMREAD_GRAYSCALE

        img0 = cv2.imread(img0_pth, imread_flag)
        img1 = cv2.imread(img1_pth, imread_flag)
        return self.from_cv_imgs(img0, img1, K0=K0, K1=K1, dist0=dist0, dist1=dist1)

    def match_images(self, image0, image1, mask0, mask1):
        batch = {'image0': image0, 'image1': image1}
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            batch.update({'mask0': ts_mask_0.unsqueeze(0), 'mask1': ts_mask_1.unsqueeze(0)})

        torch.cuda.synchronize()
        start = time.time()
        pred = self.model(batch)
        torch.cuda.synchronize()
        match_1 = time.time()
        match_time = match_1 - start

        mkpts0 = pred['keypoints0'].cpu().numpy()
        mkpts1 = pred['keypoints1'].cpu().numpy()
        matching_scores0 = pred['matching_scores'].detach().cpu().numpy()

        mconf = matching_scores0

        return mkpts0, mkpts1, mconf, match_time

    def pad_bottom_right(self, inp, pad_size, ret_mask=False):
        assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
        mask = None
        if inp.ndim == 2:
            padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
            padded[:inp.shape[0], :inp.shape[1]] = inp
            if ret_mask:
                mask = np.zeros((pad_size, pad_size), dtype=bool)
                mask[:inp.shape[0], :inp.shape[1]] = True
        elif inp.ndim == 3:
            padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
            padded[:, :inp.shape[1], :inp.shape[2]] = inp
            if ret_mask:
                mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
                mask[:, :inp.shape[1], :inp.shape[2]] = True
        else:
            raise NotImplementedError()
        return padded, mask
