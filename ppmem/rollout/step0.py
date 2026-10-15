"""Step 0: frame-0 depth (supplied or from MoGe) -> projection -> first-chunk latents."""

import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image

from ppmem.pool.init_geometry import aspect_crop_box
from ppmem.pool.projector import reproject_to_target_views
from ppmem.rollout.moge import MoGeEstimator
from ppmem.pool.layout import ensure_pool_dirs, save_step0_state
from ppmem._logging import logger


def step0(
    image_path,
    extrinsic_c2w,
    prompt,
    incremental_dir,
    h_tgt=288,
    w_tgt=512,
    device='cuda',
    moge=None,
    encoder=None,
    num_context=5,
    segment_size=25,
    init_depth=None,
    init_K=None,
):
    """
    Step 0 with latent generation:
    1. Obtain frame-0 depth and camera intrinsics, from --init_geometry or MoGe
    2. Project from frame 0 to frames 1..segment_size-1
    3. Merge: frame 0 (original) + projected frames = segment_size views
    4. Encode to latent space

    Args:
        image_path: Path to input image
        extrinsic_c2w: (N, 4, 4) camera extrinsics c2w (N >= segment_size)
        prompt: Text prompt
        incremental_dir: Pool root (state -> pool/step0_state.pt)
        h_tgt: Target height
        w_tgt: Target width
        device: Device
        moge: Optional MoGeEstimator instance (reuse to avoid reload). If None, created internally.
        encoder: LatentEncoder (required).
        segment_size: Frames per AR chunk (25)
        init_depth: optional (H, W) float32 frame-0 depth from --init_geometry, at the
            video resolution. When given, step 0 uses it for the condition projection and
            MoGe is skipped.
        init_K: the (3, 3) that describes init_depth at its own resolution. Supplied with
            init_depth or not at all -- the loader reads both from one file precisely so
            they cannot disagree.

    Returns:
        results: dict with latents and metadata (segment_size views)
    """
    pool_dir = ensure_pool_dirs(Path(incremental_dir))

    if extrinsic_c2w.shape[0] < segment_size:
        raise ValueError(f"Need at least {segment_size} frames in extrinsic, got {extrinsic_c2w.shape[0]}")

    print(f"Loading image: {image_path}")
    first_frame_pil = Image.open(image_path).convert('RGB')

    # Centre-crop to the target aspect ratio before resizing. All downstream RGB,
    # depth and intrinsics transformations then refer to the same image region.
    w_src, h_src = first_frame_pil.size
    crop_w, crop_h = aspect_crop_box(w_src, h_src, h_tgt, w_tgt)
    if (crop_w, crop_h) != (w_src, h_src):
        x0, y0 = (w_src - crop_w) // 2, (h_src - crop_h) // 2
        first_frame_pil = first_frame_pil.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        logger.info(f"Cropped image {w_src}x{h_src} -> {crop_w}x{crop_h} "
                    f"({w_tgt}:{h_tgt}) before resizing; the edges outside that are dropped")
    rgb_image = np.array(first_frame_pil)

    # Step 1: Depth and intrinsics from --init_geometry or MoGe estimation.
    original_n_frames = None
    if init_depth is not None:
        print("\n[step 0: depth] Using the depth and intrinsics from --init_geometry")
        depth = init_depth  # (H, W) at video resolution, checked by the loader
        # K matches the stored depth resolution; no rescaling is needed.
        intrinsic = np.asarray(init_K, dtype=np.float64)
        # Pad short sequences (repeat last pose + last depth)
        if extrinsic_c2w.shape[0] < segment_size:
            original_n_frames = extrinsic_c2w.shape[0]
            pad_n = segment_size - original_n_frames
            extrinsic_c2w = np.concatenate([extrinsic_c2w, np.repeat(extrinsic_c2w[-1:], pad_n, axis=0)], axis=0)
            # Only the poses are padded: frame 0's depth is the only depth step 0 reads.
            logger.warning(f"Padded short sequence: {original_n_frames} -> {segment_size} frames")
        logger.info(f"Supplied depth range: [{depth.min():.3f}, {depth.max():.3f}]")
        print(f"Supplied intrinsic fx={intrinsic[0,0]:.1f} fy={intrinsic[1,1]:.1f} cx={intrinsic[0,2]:.1f} cy={intrinsic[1,2]:.1f}")
    else:
        print("\n[step 0: depth] MoGe depth and intrinsic estimation...")
        if moge is None:
            moge = MoGeEstimator(model_type='moge-2-vitl-normal', device=device)
        depth, _, intrinsic = moge.estimate(rgb_image)
        logger.info(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
        print(f"Estimated intrinsic fx={intrinsic[0,0]:.1f} fy={intrinsic[1,1]:.1f} cx={intrinsic[0,2]:.1f} cy={intrinsic[1,2]:.1f}")

    # Step 2: Projection (RGB + Depth)
    # For step 0: project from frame 0 to frames 1..segment_size-1
    # Total: frame 0 (original) + projected frames = segment_size views
    print(f"\n[step 0: projection] Projecting to frames 1-{segment_size-1} ({segment_size-1} views)...")
    src_pose = extrinsic_c2w[0]
    tgt_poses = extrinsic_c2w[1:segment_size]

    # With --init_geometry the depth/intrinsics are at video resolution (h_tgt x w_tgt),
    # so resize rgb_image to match before projection
    proj_rgb_input = rgb_image
    proj_depth_input = depth
    if init_depth is not None and (rgb_image.shape[0] != depth.shape[0] or rgb_image.shape[1] != depth.shape[1]):
        proj_rgb_input = cv2.resize(rgb_image, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        logger.info(f"Resized image {rgb_image.shape[:2]} -> {proj_rgb_input.shape[:2]} to match the supplied depth")

    rgb_proj, mask_proj, depth_proj = reproject_to_target_views(
        src_rgb=proj_rgb_input,
        src_depth=proj_depth_input,
        src_pose=src_pose,
        tgt_poses=tgt_poses,
        intrinsic=intrinsic,
        device=device,
        return_depth=True
    )

    n_proj = rgb_proj.shape[0]  # segment_size - 1 projected views

    # Step 3: Encode to latent
    # Prepare segment_size views: frame 0 (original) + projected frames
    print(f"\n[step 0: latents] Encoding {segment_size} views to latent space...")
    assert encoder is not None, "encoder is required: LatentEncoder expects the caller to pass a VAE"

    # The prompt is not encoded here: the engine encodes it once for all chunks.

    # Resize images to target resolution
    # Frame 0: original image
    # Frames 1-24: projected images
    # Resize+crop input image to preserve aspect ratio
    h_src, w_src = rgb_image.shape[:2]
    scale = max(h_tgt / h_src, w_tgt / w_src)
    h_resized, w_resized = int(h_src * scale + 0.5), int(w_src * scale + 0.5)
    rgb_input_resized = cv2.resize(rgb_image, (w_resized, h_resized), interpolation=cv2.INTER_LINEAR)
    y0 = (h_resized - h_tgt) // 2
    x0 = (w_resized - w_tgt) // 2
    rgb_input_resized = rgb_input_resized[y0:y0+h_tgt, x0:x0+w_tgt]

    rgb_resized = np.concatenate([
        rgb_input_resized[None, ...],  # (1, H, W, 3) - frame 0
        np.stack([
            cv2.resize(rgb_proj[i], (w_tgt, h_tgt), interpolation=cv2.INTER_LINEAR)
            for i in range(n_proj)
        ], axis=0)  # (24, H, W, 3) - frames 1-24
    ], axis=0)  # (segment_size, H, W, 3)

    # For mask: frame 0 has full coverage (all ones)
    mask_frame0 = np.ones((h_tgt, w_tgt), dtype=np.float32)
    mask_resized = np.concatenate([
        mask_frame0[None, ...],  # (1, H, W) - frame 0
        np.stack([
            cv2.resize(mask_proj[i], (w_tgt, h_tgt), interpolation=cv2.INTER_LINEAR)
            for i in range(n_proj)
        ], axis=0)  # (24, H, W) - frames 1-24
    ], axis=0)  # (segment_size, H, W)

    # Mask [0,1] -> [-1,1] so VAE gets same range as RGB (encoder will use keys_no_normalize for mask)
    mask_resized_m11 = (mask_resized.astype(np.float32) * 2.0 - 1.0)  # (segment_size, H, W)
    mask_resized_3ch = np.repeat(mask_resized_m11[..., None], 3, axis=-1)  # (segment_size, H, W, 3)

    # Prepare image dict (need proj_rgb_1/2 and proj_mask_1/2 for model)
    # segment_size views total: frame 0 (original) + projected frames
    # RGB: [0,255], mask: [-1,1]
    images_dict = {
        'proj_rgb_1': rgb_resized,  # (segment_size, H, W, 3) [0,255]
        'proj_rgb_2': rgb_resized,  # can be same for first segment
        'proj_mask_1': mask_resized_3ch,  # (segment_size, H, W, 3) [-1,1]
        'proj_mask_2': mask_resized_3ch,  # (segment_size, H, W, 3) [-1,1]
    }

    # Context frames: use input image repeated num_context times
    if num_context > 0:
        context_frames = np.stack([rgb_input_resized] * num_context, axis=0)  # (N, H, W, 3)
        images_dict['context_frames'] = context_frames

    # Encode images: RGB [0,255]->[-1,1] via normalize=True; mask already [-1,1], skip normalize
    latents_dict = encoder.encode_images_batch(
        images_dict, normalize=True, keys_no_normalize=['proj_mask_1', 'proj_mask_2'],
    )

    latents_dict = encoder.rearrange_latents(latents_dict)

    latent = latents_dict['proj_rgb_1']  # (B, F*H*W, C)
    B, FHW, C = latent.shape

    # Latent dimensions: spatial downscale 16; the separate VAE is 1:1 per frame,
    # so latent frames == segment_size.
    latent_h = h_tgt // 16
    latent_w = w_tgt // 16
    latent_f = segment_size

    if FHW != latent_f * latent_h * latent_w:
        raise ValueError(
            f"proj_rgb_1 latent has {FHW} patches but expected "
            f"F*H*W = {latent_f}*{latent_h}*{latent_w} = {latent_f * latent_h * latent_w} "
            f"(segment_size={segment_size})"
        )

    # Scale intrinsics from original image size to video size (h_tgt, w_tgt)
    # MoGe intrinsic is for rgb_image shape; a supplied one is already at video resolution
    if init_depth is not None:
        fx_video = float(intrinsic[0, 0])
        fy_video = float(intrinsic[1, 1])
        cx_video = float(intrinsic[0, 2])
        cy_video = float(intrinsic[1, 2])
    else:
        H_orig, W_orig = rgb_image.shape[:2]
        scale_x = w_tgt / float(W_orig)
        scale_y = h_tgt / float(H_orig)
        fx_video = float(intrinsic[0, 0]) * scale_x
        fy_video = float(intrinsic[1, 1]) * scale_y
        cx_video = float(intrinsic[0, 2]) * scale_x
        cy_video = float(intrinsic[1, 2]) * scale_y

    # Prepare intrinsics list (one per frame) at video resolution
    intrinsics_list = []
    for i in range(segment_size):
        intrinsics_list.append({
            'fx': fx_video,
            'fy': fy_video,
            'cx': cx_video,
            'cy': cy_video,
        })

    # Prepare next_ar_depth: use projected depth for frames 1-24
    # Frame 0 uses the frame-0 depth (supplied or from MoGe)
    depth_frame0 = cv2.resize(depth, (w_tgt, h_tgt), interpolation=cv2.INTER_LINEAR)

    depth_proj_resized = np.stack([
        cv2.resize(depth_proj[i], (w_tgt, h_tgt), interpolation=cv2.INTER_LINEAR)
        for i in range(n_proj)
    ], axis=0)  # (24, H, W)

    next_ar_depth = torch.from_numpy(np.concatenate([
        depth_frame0[None, ...],  # (1, H, W) - frame 0
        depth_proj_resized  # (segment_size-1, H, W) - projected frames
    ], axis=0))  # (segment_size, H, W)

    # Prepare context_depth
    if num_context > 0:
        context_depth = torch.from_numpy(np.stack([depth_frame0] * num_context, axis=0))  # (N, H, W)
    else:
        context_depth = torch.zeros(0, h_tgt, w_tgt)

    # Build metadata (full trajectory so step1+ and next_ar_depth can use it; step=0 → segment [0..segment_size))
    meta_info = {
        'name': Path(image_path).stem,
        'text': prompt,
        'step': 0,
        'video_num_frames': segment_size,  # views per segment
        'original_n_frames': original_n_frames,  # None if no padding, else real frame count
        'segment_stride': segment_size - 1,  # pose advance per AR step (overlap by 1)
        'video_height': h_tgt,
        'video_width': w_tgt,
        'latent_num_frames': latent_f,
        'latent_height': latent_h,
        'latent_width': latent_w,
        'context_num_frames': num_context,
        'intrinsics': intrinsics_list,
        'extrinsics_c2w': extrinsic_c2w.tolist(),  # Full trajectory (needs >= segment_size + stride poses)
        'next_ar_depth': next_ar_depth,  # (segment_size, H, W)
        'context_depth': context_depth,  # (N, H, W)
    }

    for name in images_dict.keys():
        meta_info[f'latent_{name}'] = latents_dict[name].cpu().detach()[0]

    # Add very_first_frame (VF) and history
    vf_images = {'very_first_frame': np.stack([rgb_input_resized], axis=0)}  # (1, H, W, 3)
    vf_latents = encoder.encode_images_batch(vf_images, normalize=True)
    vf_latents = encoder.rearrange_latents(vf_latents)
    meta_info['latent_very_first_frame'] = vf_latents['very_first_frame'].cpu().detach()[0]
    meta_info['very_first_num_frames'] = 1
    meta_info['latent_history'] = torch.zeros(0, C)
    meta_info['history_num_frames'] = 0

    meta_info['first_frame_pil_images'] = first_frame_pil
    # Frames 1.. of the projection, at projection resolution, for the diagnostic grid.
    meta_info['proj_rgb_frames'] = np.clip(rgb_proj, 0, 255).astype(np.uint8)

    # Later chunks and resume only need a few fields; the rest goes to generate_video in memory.
    save_step0_state(pool_dir, meta_info)

    return meta_info
