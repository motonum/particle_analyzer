import os

class Config:
    """設定値を管理するクラス"""
    # --- 物理パラメータ ---
    DOTS_PER_MICRON = 1024 / 212.13
    SLICE_STEP_MICRON = 2.0

    # --- セグメンテーションパラメータ ---
    GAUSSIAN_BLUR_KERNEL = (5, 5)
    MORPHOLOGY_KERNEL_SIZE = (3, 3)
    FILL_HOLES_STRUCTURE_SIZE = (7, 7)
    MORPHOLOGY_ITERATIONS = 2
    PEAK_MIN_DISTANCE_RATIO = 1.0  # マーカー間の最小距離（半径に対する比率）
    PEAK_THRESHOLD_RATIO = 1.0     # マーカー検出の閾値（半径に対する比率）

    # --- 粒子フィルタリングパラメータ ---
    MIN_DIAMETER_MICRON = 1.0
    MIN_CIRCULARITY = 0.65

    # --- 出力設定 ---
    HISTOGRAM_BINS_PER_MICRON = 2