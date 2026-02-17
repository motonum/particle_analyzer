import numpy as np
from enum import Enum
import math


class Stats:
    """統計量を計算するクラス"""

    class StatsItem(Enum):
        COUNT = "Count"
        MEAN = "Mean"
        MEDIAN = "Median"
        STD_DEVIATION = "Standard Deviation"
        MIN = "Min"
        MAX = "Max"
        IQR1 = "IQR1"
        IQR3 = "IQR3"
        CV = "CV"

    @staticmethod
    def mean(data: list[float]) -> float:
        """平均値を計算する"""
        return np.mean(data) if data else 0.0

    @staticmethod
    def std_deviation(data: list[float]) -> float:
        """標準偏差を計算する"""
        return np.std(data, ddof=1) if len(data) > 1 else 0.0

    @staticmethod
    def median(data: list[float]) -> float:
        """中央値を計算する"""
        return np.median(data) if data else 0.0

    @staticmethod
    def min(data: list[float]) -> float:
        """最小値を計算する"""
        return np.min(data) if data else 0.0

    @staticmethod
    def max(data: list[float]) -> float:
        """最大値を計算する"""
        return np.max(data) if data else 0.0

    @staticmethod
    def count(data: list[float]) -> int:
        """データの個数を計算する"""
        return len(data)

    @staticmethod
    def percentile(data: list[float], percent: float) -> float:
        """パーセンタイルを計算する"""
        return np.percentile(data, percent) if data else 0.0

    @staticmethod
    def iqr1(data: list[float]) -> float:
        """パーセンタイルを計算する"""
        return np.percentile(data, 25)

    def iqr3(data: list[float]) -> float:
        """パーセンタイルを計算する"""
        return np.percentile(data, 75)

    @staticmethod
    def frequency_within_range(data: list[float], min_val: float | None, max_val: float | None) -> int:
        """指定した範囲内のデータの個数を計算する"""
        if min_val is None and max_val is None:
            return len(data)
        elif min_val is None:
            return len([d for d in data if d <= max_val])
        elif max_val is None:
            return len([d for d in data if d >= min_val])
        else:
            return len([d for d in data if min_val <= d <= max_val])

    @staticmethod
    def frequency_percentage_within_range(data: list[float], min_val: float | None, max_val: float | None) -> float:
        """指定した範囲内のデータの割合を計算する"""
        count = Stats.frequency_within_range(data, min_val, max_val)
        return (count / len(data) * 100) if data else 0.0

    @staticmethod
    def mode_bin(
        data: list[float],
        bin_width: float,
        density: bool = False,
    ) -> tuple[float, float]:
        """粒子の最頻値直径を取得する

        Parameters
        ----------
        data: list[float]
            データのリスト
        bin_width: float = None
            ヒストグラムのビン幅
        density: bool = False
            Trueのとき、ヒストグラムの縦軸を相対度数とする

        Returns
        -------
        quantity: float
            最頻値の度数
        bin_range: (float, float)
            最頻値のビンの範囲
        """
        if bin_width <= 0:
            raise ValueError("bin_width must be greater than 0")
        if not data:
            return 0.0, 0.0
        bins = math.ceil(max(data) / bin_width) + 1
        range = (0.0, bin_width * bins)
        print(bins, range)
        hist, _ = np.histogram(data, bins=bins, density=density, range=range)
        mode_index = np.argmax(hist)
        bin_range = (bin_width * mode_index, bin_width * (mode_index + 1))
        quantity: float = hist[mode_index]
        return quantity, bin_range

    @staticmethod
    def cv(data: list[float]):
        """変動係数を計算する"""
        if not data:
            return 0.0
        mean = Stats.mean(data)
        if mean == 0:
            return 0.0
        return Stats.std_deviation(data) / mean

    @staticmethod
    def summary_statistics(data: list[float]) -> dict[StatsItem, float]:
        """データの基本統計量を計算する"""
        return {
            Stats.StatsItem.COUNT: Stats.count(data),
            Stats.StatsItem.MEAN: Stats.mean(data),
            Stats.StatsItem.MEDIAN: Stats.median(data),
            Stats.StatsItem.STD_DEVIATION: Stats.std_deviation(data),
            Stats.StatsItem.MIN: Stats.min(data),
            Stats.StatsItem.MAX: Stats.max(data),
            Stats.StatsItem.IQR1: Stats.iqr1(data),
            Stats.StatsItem.IQR3: Stats.iqr3(data),
            Stats.StatsItem.CV: Stats.cv(data),
        }

    # 統計量のenumの一覧のリストを取得する
    @staticmethod
    def get_stats_items() -> list[Enum]:
        return list(Stats.StatsItem)
