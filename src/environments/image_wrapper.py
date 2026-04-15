import gymnasium as gym
import numpy as np
import cv2
from gymnasium.wrappers import AddRenderObservation, FrameStackObservation

# Crops como fracciones del frame (top, bottom, left, right)
CROPS: tuple[float, float, float, float] = (0.25, 0.05, 0.10, 0.10)


def _crop_frame_hwc(
    frame: np.ndarray, crops: tuple[float, float, float, float] = CROPS
) -> np.ndarray:
    """
    Crop de un frame RGB HxWx3 usando ratios.
    Devuelve frame recortado (si el crop fuera inválido, devuelve el original).
    """
    if frame is None or frame.size == 0:
        return frame

    top_f, bottom_f, left_f, right_f = crops
    h, w = frame.shape[0], frame.shape[1]

    top_px = int(round(h * top_f))
    bottom_px = int(round(h * bottom_f))
    left_px = int(round(w * left_f))
    right_px = int(round(w * right_f))

    y0 = max(0, top_px)
    y1 = min(h, h - max(0, bottom_px))
    x0 = max(0, left_px)
    x1 = min(w, w - max(0, right_px))

    if y1 <= y0 or x1 <= x0:
        return frame

    return frame[y0:y1, x0:x1, :]


def make_pixel_env(env: gym.Env, frame_stack: int = 4) -> gym.Env:
    """
    1) AddRenderObservation(render_only=True) -> obs: RGB (H,W,3)
    2) FrameStackObservation -> obs: (S,H,W,3)
    """
    env = AddRenderObservation(env, render_only=True)
    env = FrameStackObservation(env, stack_size=frame_stack)
    return env


def obs_to_uint8_hwc_stacked(
    obs,
    image_size: int = 84,
    crops: tuple[float, float, float, float] = CROPS,
) -> np.ndarray:
    """
    Convierte obs apilada a uint8 en formato HWC con canales concatenados:
      - Entrada típica: (S,H,W,3)
      - Salida: (image_size, image_size, 3*S)

    Mantiene COLOR (RGB) para distinguir piernas.
    """
    x = np.array(obs)

    # Caso (S,H,W,3)
    if x.ndim == 4 and x.shape[-1] == 3:
        frames = []
        for i in range(x.shape[0]):
            f = x[i]  # (H,W,3) RGB
            f = _crop_frame_hwc(f, crops=crops)
            f = cv2.resize(f, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frames.append(f)

        stacked = np.concatenate(frames, axis=-1)  # (H,W,3*S)
        if stacked.dtype != np.uint8:
            stacked = np.clip(stacked, 0, 255).astype(np.uint8)
        return stacked

    # Caso (H,W,3*S) ya apilado
    if x.ndim == 3 and x.shape[-1] % 3 == 0 and x.shape[-1] != 3:
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)
        # (Opcional) si quisieras aplicar crop+resize aquí también, habría que separar frames; lo normal es no hacerlo.
        return x

    # Caso (H,W,3) un frame
    if x.ndim == 3 and x.shape[-1] == 3:
        x = _crop_frame_hwc(x, crops=crops)
        x = cv2.resize(x, (image_size, image_size), interpolation=cv2.INTER_AREA)
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)
        return x

    raise ValueError(f"Unsupported observation shape for pixel conversion: {x.shape}")
