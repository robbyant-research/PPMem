"""MoGe Depth Estimation Module"""

import torch
import numpy as np
from ppmem.checkpoints import moge_source


# MoGe is imported inside MoGeEstimator.__init__, not at module level, so that
# `import ppmem.rollout` works without MoGe installed (--init_geometry doesn't need it).
MODEL_TYPE = {
    'moge-2-vitl-normal': {
        'has_normal': True,
    },
}


class MoGeEstimator:
    """MoGe depth and normal estimator"""

    def __init__(self, model_type='moge-2-vitl-normal', device='cuda', checkpoint=None):
        """
        Initialize MoGe estimator
        
        Args:
            model_type: only 'moge-2-vitl-normal'; anything else raises KeyError.
            device: Device to run on
            checkpoint: Local MoGe weight file or Hugging Face repo ID.
        """
        self.device = device
        self.has_normal = MODEL_TYPE[model_type]['has_normal']

        try:
            from moge.model.v2 import MoGeModel
        except ImportError as e:
            raise ImportError(
                "The MoGe depth path needs MoGe v2, which could not be imported. Two options:\n"
                "  1. follow README's environment setup, then from the PPMem repository:\n"
                "     python -m pip install --no-build-isolation -r requirements.txt\n"
                "  2. check out the source and put it on PYTHONPATH: "
                "export PYTHONPATH=\"$REPO:/path/to/MoGe\"\n"
                "     Point at the MoGe directory itself, not its parent: the utils3d there\n"
                "     shadows the installed one, imports still succeed and inference fails later.\n"
                "With a depth map in hand, --init_geometry does not need MoGe at all."
            ) from e

        print(f"Loading MoGe model: {model_type}")
        self.model = MoGeModel.from_pretrained(
            moge_source(checkpoint)
        ).to(device)
        self.model.eval()

    def estimate(self, rgb_image, intrinsic=None):
        """
        Estimate depth and normal from RGB image. Optionally estimate camera intrinsics.
        
        Args:
            rgb_image: (H, W, 3) numpy array, uint8 [0-255] or float [0-1]
            intrinsic: (3, 3) numpy array, optional. If None, MoGe estimates intrinsics from the image.
        
        Returns:
            depth: (H, W) numpy array, metric depth
            normal: (3, H, W) numpy array, normal map in [0, 1] range
            intrinsic: (3, 3) numpy array, camera intrinsic matrix (pixel space). From input if provided, else from MoGe output.
        """
        # Normalize input
        if rgb_image.dtype == np.uint8:
            rgb_normalized = rgb_image.astype(np.float32) / 255.0
        else:
            rgb_normalized = rgb_image.astype(np.float32)
            if rgb_normalized.max() > 1.0:
                rgb_normalized = rgb_normalized / 255.0

        # Convert to tensor: (H, W, 3) -> (3, H, W)
        input_tensor = torch.tensor(
            rgb_normalized, dtype=torch.float32, device=self.device
        ).permute(2, 0, 1)

        h, w = rgb_image.shape[:2]

        # Inference: optional fov_x if intrinsic provided (can improve accuracy)
        with torch.no_grad():
            if intrinsic is not None:
                fx = intrinsic[0, 0]
                fov_x = 2.0 * np.arctan(w / (2.0 * fx)) * 180.0 / np.pi
                output = self.model.infer(input_tensor, fov_x=fov_x)
                out_intrinsic = np.array(intrinsic, dtype=np.float32)
            else:
                output = self.model.infer(input_tensor)
                # MoGe returns normalized intrinsics (3,3); convert to pixel space
                K_norm = output.get('intrinsics')
                if K_norm is None:
                    raise RuntimeError("MoGe output has no 'intrinsics'; cannot estimate camera.")
                if isinstance(K_norm, torch.Tensor):
                    K_norm = K_norm.cpu().numpy()
                K_norm = np.array(K_norm, dtype=np.float64)
                # Normalized convention: fx_norm = fx/w, cx_norm = cx/w, fy_norm = fy/h, cy_norm = cy/h
                out_intrinsic = np.array([
                    [K_norm[0, 0] * w, K_norm[0, 1], K_norm[0, 2] * w],
                    [K_norm[1, 0], K_norm[1, 1] * h, K_norm[1, 2] * h],
                    [K_norm[2, 0], K_norm[2, 1], K_norm[2, 2]],
                ], dtype=np.float32)

        # Extract depth
        pred_depth = output['depth']
        if isinstance(pred_depth, torch.Tensor):
            pred_depth = pred_depth.cpu().numpy()

        # Extract normal
        if self.has_normal and 'normal' in output and output['normal'] is not None:
            pred_normal = output['normal']
            if isinstance(pred_normal, torch.Tensor):
                pred_normal = pred_normal.cpu().numpy()
            pred_normal_normalized = (pred_normal + 1.0) / 2.0
            pred_normal_normalized = np.clip(pred_normal_normalized, 0.0, 1.0)
        else:
            h, w = pred_depth.shape
            pred_normal_normalized = np.zeros((h, w, 3), dtype=np.float32)

        # MoGe writes inf at pixels it cannot resolve, and the mask below is 0 there.
        # inf * 0 is nan, which the clip does not remove, so it reaches the plane pool
        # as a plane with no position. Clear the non-finite values before masking.
        pred_depth = np.nan_to_num(pred_depth, nan=0.0, posinf=0.0, neginf=0.0)
        pred_normal_normalized = np.nan_to_num(
            pred_normal_normalized, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply mask if available
        if 'mask' in output and output['mask'] is not None:
            mask = output['mask']
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            pred_depth = pred_depth * mask
            pred_normal_normalized = pred_normal_normalized * mask[..., None]

        # Clamp depth
        pred_depth = np.clip(pred_depth, 0, 300)
        # Convert normal to (3, H, W)
        pred_normal_normalized = pred_normal_normalized.transpose(2, 0, 1)

        return pred_depth, pred_normal_normalized, out_intrinsic
