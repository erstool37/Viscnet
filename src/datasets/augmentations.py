"""Config-driven, clip-consistent video augmentation helpers."""

from __future__ import annotations

import albumentations as A
import cv2


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def _scaled_probability(base_probability, policy_probability):
    return _clamp(float(base_probability) * float(policy_probability), 0.0, 1.0)


def _odd_blur_limit(base_limit, strength):
    limit = max(3, int(round(float(base_limit) * float(strength))))
    if limit % 2 == 0:
        limit += 1
    return limit


def build_video_augmentation(config=None, output_size=224):
    """Build an albumentations ReplayCompose for one video clip.

    The returned transform is meant to be sampled once per clip, then replayed
    across the remaining frames so geometry and photometric jitter stay
    temporally consistent.
    """

    config = dict(config or {})
    policy_name = str(config.get("type", config.get("policy", "legacy"))).lower()
    probability = _clamp(float(config.get("probability", 1.0)), 0.0, 1.0)
    strength = max(0.0, float(config.get("strength", 1.0)))

    transforms = []
    if output_size is not None:
        transforms.append(A.Resize(int(output_size), int(output_size), interpolation=cv2.INTER_LINEAR, p=1.0))

    if policy_name == "legacy":
        transforms.extend(
            [
                A.Perspective(scale=(0.01, 0.02), keep_size=True, p=_scaled_probability(0.6, probability)),
                A.MotionBlur(blur_limit=(3, 7), p=_scaled_probability(0.6, probability)),
                A.RandomBrightnessContrast(0.05, 0.1, p=_scaled_probability(0.5, probability)),
            ]
        )
    elif policy_name == "augv1":
        transforms.extend(
            [
                A.Perspective(
                    scale=(0.005 * strength, 0.02 * strength),
                    keep_size=True,
                    p=_scaled_probability(0.5, probability),
                ),
                A.MotionBlur(
                    blur_limit=(3, _odd_blur_limit(5, strength)),
                    p=_scaled_probability(0.35, probability),
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.04 * strength,
                    contrast_limit=0.08 * strength,
                    p=_scaled_probability(0.55, probability),
                ),
            ]
        )
    elif policy_name == "augv2":
        transforms.extend(
            [
                A.Perspective(
                    scale=(0.01 * strength, 0.035 * strength),
                    keep_size=True,
                    p=_scaled_probability(0.65, probability),
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.02 * strength,
                    scale_limit=0.05 * strength,
                    rotate_limit=4.0 * strength,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=_scaled_probability(0.6, probability),
                ),
                A.MotionBlur(
                    blur_limit=(3, _odd_blur_limit(7, strength)),
                    p=_scaled_probability(0.5, probability),
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.08 * strength,
                    contrast_limit=0.15 * strength,
                    p=_scaled_probability(0.7, probability),
                ),
            ]
        )
    elif policy_name == "augv2_noise":
        transforms.extend(
            [
                A.Perspective(
                    scale=(0.01 * strength, 0.035 * strength),
                    keep_size=True,
                    p=_scaled_probability(0.65, probability),
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.02 * strength,
                    scale_limit=0.05 * strength,
                    rotate_limit=4.0 * strength,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=_scaled_probability(0.6, probability),
                ),
                A.MotionBlur(
                    blur_limit=(3, _odd_blur_limit(7, strength)),
                    p=_scaled_probability(0.5, probability),
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.08 * strength,
                    contrast_limit=0.15 * strength,
                    p=_scaled_probability(0.7, probability),
                ),
                A.GaussNoise(
                    var_limit=(5.0, 25.0),
                    mean=0,
                    per_channel=True,
                    p=_scaled_probability(0.35, probability),
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported video augmentation policy: {policy_name}")

    transform = A.ReplayCompose(transforms)
    transform.policy_name = policy_name
    transform.probability = probability
    transform.strength = strength
    return transform


def apply_video_augmentation_consistently(frames, transform):
    """Apply one sampled augmentation replay across every frame in a clip."""

    if not frames:
        return []

    first = transform(image=frames[0])
    replay = first.get("replay")
    if replay is None:
        raise ValueError("Video augmentation transform must provide replay metadata")

    augmented = [first["image"]]
    for frame in frames[1:]:
        augmented.append(A.ReplayCompose.replay(replay, image=frame)["image"])
    return augmented
