import torch
import numpy as np
from typing import List, Dict, Tuple
from segmentation import Segment

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Register = Dict[str, torch.Tensor] | None


def svd(
    feature: torch.Tensor, regularize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _feature = feature.clone()
    mean = _feature.mean(dim=1, keepdim=True)
    _feature -= mean
    centered = _feature  # Don't modify original
    num_samples = feature.size(1)
    if num_samples == 1:
        covariant = torch.mm(centered, centered.t())
    else:
        covariant = torch.mm(centered, centered.t()) / (num_samples - 1)

    if regularize:
        covariant += torch.eye(covariant.size(0), device=DEVICE)

    output = torch.svd(covariant, some=False)
    return output


def nchw_to_c_hw(feature: torch.Tensor) -> torch.Tensor:
    feature = feature.squeeze(0)
    c = feature.size(0)
    return feature.view(c, -1).clone()


def get_rank(singulars: torch.Tensor, dim: int, eps: float = 1e-5) -> int:
    rank = dim
    for i in range(dim - 1, -1, -1):
        if singulars[i] >= eps:
            rank = i + 1
            break
    return rank


def wct(
    content_feature: torch.Tensor,
    style_feature: torch.Tensor,
    alpha: float = 1.0,
    registers: Register = None,
) -> torch.Tensor:
    if not registers:
        style_feature = nchw_to_c_hw(style_feature)
        style_mean = torch.mean(style_feature, 1)
        _, style_e_vals, style_e_vecs = svd(style_feature, regularize=True)
        style_rank = get_rank(style_e_vals, style_feature.shape[0])
        style_e_vals_sqrt = style_e_vals[:style_rank].pow(0.5)
        style_e_vecs_truncated = style_e_vecs[:, :style_rank]
        coloring_matrix = torch.mm(
            torch.mm(style_e_vecs_truncated, torch.diag(style_e_vals_sqrt) * alpha),
            style_e_vecs_truncated.t(),
        )
        registers = {
            "coloring_matrix": coloring_matrix,
            "style_mean": style_mean,
        }
    else:
        coloring_matrix = registers["coloring_matrix"]
        style_mean = registers["style_mean"]

    content_feature = nchw_to_c_hw(content_feature)
    content_min, content_max = content_feature.min(), content_feature.max()
    content_mean = content_feature.mean(dim=1, keepdim=True)
    # KEY FIX: Clone before modifying to avoid in-place modification
    _content_feature = content_feature.clone()
    _content_feature -= content_mean

    _, content_e_vals, content_e_vecs = svd(_content_feature, regularize=True)
    content_rank = get_rank(content_e_vals, _content_feature.shape[0])
    content_e_vals_invsqrt = content_e_vals[:content_rank].pow(-0.5)
    content_e_vecs_truncated = content_e_vecs[:, :content_rank]

    scaled = content_e_vecs_truncated * content_e_vals_invsqrt.unsqueeze(0)
    whitened_content = torch.mm(
        torch.mm(scaled, content_e_vecs_truncated.t()), _content_feature
    )
    colored_content = torch.mm(coloring_matrix, whitened_content)
    colored_content += style_mean.unsqueeze(1).expand_as(colored_content)
    colored_content = colored_content.clamp(content_min, content_max)
    return colored_content


def _resize_mask(mask: np.ndarray, target_size: torch.Size) -> torch.Tensor:
    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=DEVICE)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    mask_tensor = torch.nn.functional.interpolate(
        mask_tensor, target_size, mode="nearest"
    )
    mask_tensor = mask_tensor.squeeze().flatten()
    return mask_tensor


def wct_segment_new(
    content_feature: torch.Tensor,
    style_feature: torch.Tensor,
    content_segment: List[Segment],
    style_segment: List[Segment],
    alpha: float = 1.0,
    registers: Register = None,
) -> torch.Tensor:
    content_feature_size = content_feature.shape[-2:]
    style_feature_size = style_feature.shape[-2:]

    content_feature = nchw_to_c_hw(content_feature)
    style_feature = nchw_to_c_hw(style_feature)

    content_masks = {mask.id: mask for mask in content_segment}
    style_masks = {mask.id: mask for mask in style_segment}

    compatible_masks_ids = set(content_masks.keys()).intersection(
        set(style_masks.keys())
    )

    result = content_feature.clone()

    for mask_id in compatible_masks_ids:
        content_mask = _resize_mask(content_masks[mask_id].map, content_feature_size)
        style_mask = _resize_mask(style_masks[mask_id].map, style_feature_size)
        content_indices = content_mask.nonzero().squeeze(-1)
        style_indices = style_mask.nonzero().squeeze(-1)

        content_segment_target = content_feature[:, content_indices]
        style_segment_target = style_feature[:, style_indices]

        stylized_segment = wct(
            content_segment_target,
            style_segment_target,
            alpha=alpha,
            registers=registers,
        )
        result[:, content_indices] = stylized_segment

    return result


@torch.compile(mode="max-autotune")
def perform_wct(
    content_feature: torch.Tensor,
    style_feature: torch.Tensor,
    content_masks: List[Segment] | None = None,
    style_masks: List[Segment] | None = None,
    weight: float = 1.0,
    registers: Register = None,
    alpha: float = 1.0,
    device: str = DEVICE,
) -> torch.Tensor:
    # Clone content_feature to avoid any possibility of in-place modification
    content_feature = content_feature.clone()

    if content_masks is not None and style_masks is not None:
        result = wct_segment_new(
            content_feature,
            style_feature,
            content_masks,
            style_masks,
            weight,
            registers,
        )

    else:
        result = wct(content_feature, style_feature, alpha=weight, registers=registers)
    result = result.view_as(content_feature)
    print("This is alpha", alpha)
    result = alpha * result + (1 - alpha) * content_feature
    return result
