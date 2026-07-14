"""
wavio.py — 共用 WAV 讀寫輔助（專案唯一寫檔入口）

寫法：float32 WAV（PCM_FLOAT，32-bit）
  - 任何播放器與 ffmpeg 按容器頭正確解讀，滿刻度 ±1.0
  - 不再有 24-bit 值塞入 int32 容器的刻度歧義

讀法：自動偵測格式，統一回傳 float64 [-1, 1]
  - float32 WAV：直接轉 float64
  - int32 WAV（舊 24-bit-in-int32）：除以 2^23 相容讀取
  - int16 WAV：除以 32768.0
"""

import numpy as np
from scipy.io import wavfile
import pathlib


def write_wav(path, sr: int, sig: np.ndarray) -> None:
    """
    將 float64/float32 訊號（±1.0 範圍）寫成 float32 WAV。

    Args:
        path: 輸出路徑（str 或 Path）
        sr:   取樣率
        sig:  float64 或 float32，shape (N,) 或 (N, ch)
    """
    data = np.clip(sig, -1.0, 1.0).astype(np.float32)
    wavfile.write(str(path), sr, data)


def load_wav(path, max_s: float | None = None, target_sr: int | None = None
             ) -> tuple[int, np.ndarray]:
    """
    讀 WAV，統一回傳 (sr, float64_mono)。

    相容性：
      - float32 WAV（本專案新格式）→ 直接轉 float64
      - int32 WAV（舊 24-bit-in-int32 格式，2^23 縮放）→ 除以 2^23
      - int32 WAV（ffmpeg pcm_s32le 全範圍）→ 除以 2^31
        ※ 區分方法：舊格式 peak 值 < 2^23，ffmpeg 格式 peak 值 > 2^23
      - int16 WAV → 除以 32768

    Args:
        path:      WAV 路徑
        max_s:     若指定，只取前 max_s 秒
        target_sr: 若指定，檢查 sr 是否符合（不符合則 raise）

    Returns:
        (sr, data_float64_mono)
    """
    sr, x = wavfile.read(str(path))

    if target_sr is not None and sr != target_sr:
        raise ValueError(f"load_wav: expected sr={target_sr}, got sr={sr} in {path}")

    if x.dtype == np.float32 or x.dtype == np.float64:
        data = x.astype(np.float64)
    elif x.dtype == np.int16:
        data = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:
        # 判斷是本專案舊 24-bit 格式還是 ffmpeg 全範圍 int32
        peak = np.max(np.abs(x))
        if peak <= (2 ** 23):
            # 舊 24-bit-in-int32，縮放因子 2^23
            data = x.astype(np.float64) / (2 ** 23)
        else:
            # ffmpeg pcm_s32le 全 32-bit 範圍
            data = x.astype(np.float64) / 2147483648.0
    else:
        data = x.astype(np.float64)

    if data.ndim > 1:
        data = data.mean(axis=1)

    if max_s is not None:
        data = data[:int(sr * max_s)]

    return sr, data.astype(np.float64)
