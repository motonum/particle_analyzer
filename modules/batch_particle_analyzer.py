import math
from tqdm import tqdm
import csv
import os

from modules.stats import Stats
from modules.particle_analyzer import ParticleAnalyzer, Config


class BatchParticleAnalyzer:
    """複数の粒子解析を一括で実行するクラス

    複数のParticleAnalyzerインスタンスを保持し、それぞれの解析や結果出力を一括で実行する。
    また、複数の解析結果をまとめて、正規化されたヒストグラムやサマリーを作成する機能も提供する。
    """

    def __init__(self, particle_analyzers: list[ParticleAnalyzer], config: Config):
        """
        Parameters
        ----------
        particle_analyzers : list[ParticleAnalyzer]
            実行対象のParticleAnalyzerインスタンスのリスト
        config : Config
            解析に使用する設定オブジェクト
        """
        self.particle_analyzers = particle_analyzers
        self.config = config

    def run_analysis(self, strict_tracking: bool = True):
        """登録されているすべてのParticleAnalyzerで解析を実行する"""
        with tqdm(
            total=len(self.particle_analyzers),
            desc="Analyzing",
            leave=True,
            bar_format=f"{{l_bar}}{{bar}} | {{n_fmt}}/{{total_fmt}}",
        ) as pbar:
            for analyzer in self.particle_analyzers:
                pbar.set_description(
                    f"Analyzing '{analyzer.image_interface.parent_dir}{analyzer.image_interface.filename}.tif'"
                )
                analyzer.run_analysis(strict_tracking=strict_tracking)
                pbar.update(1)
            pbar.set_description(f"Completed")
            pbar.close()

    def output_summary_csv(self, ranges: list[tuple[int | None, int | None]] = None, title: str = "batch_summary"):
        """全画像の統計情報をまとめて1つのCSVファイルとして出力する

        各画像の統計情報（平均値、中央値、標準偏差など）と、
        指定された範囲内の粒子の個数や割合をまとめたCSVファイルを出力する。

        Parameters
        ----------
        ranges : list[tuple[int | None, int | None]]
            集計対象とする直径の範囲を指定するタプルのリスト
            例: [(None, 10), (10, 20), (20, None)]
        filename : str = "batch_summary"
            出力するCSVファイルの名前（拡張子なし）
        """
        # 出力する統計情報の列名を準備
        stats_columns = [item.value for item in Stats.get_stats_items()]

        # 範囲指定がある場合の列名を準備
        range_columns = []
        if ranges:
            for min_val, max_val in ranges:
                range_name = (
                    f"{min_val if min_val is not None else '0'}" f"-{max_val if max_val is not None else 'inf'}" f"_μm"
                )
                range_columns.extend([f"Count_{range_name}", f"Percentage_{range_name}"])

        # ヘッダー行を作成
        headers = ["Filename"] + stats_columns + range_columns

        # 各画像の統計情報を収集
        rows = []
        for analyzer in self.particle_analyzers:
            filename = analyzer.image_interface.filename
            diameters, _ = analyzer._get_diameters()

            # 基本統計量を取得
            stats = Stats.summary_statistics(diameters)
            stats_values = [stats[item] for item in Stats.get_stats_items()]

            # 範囲指定がある場合の統計を取得
            range_values = []
            if ranges:
                for min_val, max_val in ranges:
                    count = Stats.frequency_within_range(diameters, min_val, max_val)
                    percentage = Stats.frequency_percentage_within_range(diameters, min_val, max_val)
                    range_values.extend([count, percentage])

            # 1行分のデータを作成
            row = [filename] + stats_values + range_values
            rows.append(row)

        output_dir = self.config.OUTPUT_DIR_SUMMARY
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{title}.csv")

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        print(f"Batch summary CSV saved to: {output_path}")

    def output_particle_image(self):
        """登録されているすべてのParticleAnalyzerで粒子検出画像を出力する"""
        for analyzer in self.particle_analyzers:
            analyzer.output_particle_image()

    def get_max_diameter(self, z: int | None = None, auto_z: bool = False) -> float:
        """全解析結果の中から最大の粒子直径を取得する

        Parameters
        ----------
        z : int | None = None
            直径を取得するzスライスを指定
        auto_z : bool = False
            Trueの場合、粒子が最初に現れるスライスの次のスライスを対象とする

        Returns
        -------
        float
            最大の粒子直径 (μm)
        """
        return max(analyzer.get_max_diameter(z, auto_z) for analyzer in self.particle_analyzers)

    def plot_diameter_histogram(
        self,
        add_title: bool = True,
        z: float | None = None,
        auto_z: bool = False,
        density: bool = True,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        bin_width: float | None = None,
    ):
        """全解析結果の直径ヒストグラムを、スケールを統一してプロットする

        y軸の表示範囲を全画像で統一し、比較しやすくしたヒストグラムを出力する。

        Parameters
        ----------
        add_title : bool = True
            グラフにタイトルを追加するかどうか
        z : float | None = None
            直径を取得するzスライスを指定
        auto_z : bool = False
            Trueの場合、粒子が最初に現れるスライスの次のスライスを対象とする
        density : bool = True
            Trueの場合、ヒストグラムの縦軸を相対度数とする
        xlim : tuple[float, float] | None = None
            グラフの横軸の表示範囲
        ylim : tuple[float, float] | None = None
            グラフの縦軸の表示範囲
        bin_width : float | None = None
            ヒストグラムのビン幅 (μm)
        """
        limit_diameter = self.get_max_diameter(z, auto_z) if not xlim else xlim[1]
        step = self.config.DEFAULT_HISTOGRAM_BIN_WIDTH if not bin_width else bin_width
        bins = math.ceil(limit_diameter / step)
        x_range = (0, int(step * bins) + step) if not xlim else xlim

        # ylimを各画像の最大値に合わせる場合、全画像の最大値を取得
        y_limit = (
            max([analyzer.calc_mode_bin(step, density, z, auto_z)[1] for analyzer in self.particle_analyzers])
            if ylim is None
            else ylim[1]
        )
        y_range = (0, y_limit * 1.1) if ylim is None else ylim

        for analyzer in self.particle_analyzers:
            title = analyzer.image_interface.filename if add_title else ""
            analyzer.plot_diameter_histogram(title, z, auto_z, density, x_range, y_range, bin_width)

    def output_diameter_csv(self, z: int | None = None, auto_z: bool = False, header: bool = True):
        """登録されているすべてのParticleAnalyzerで直径の生データCSVを出力する

        Parameters
        ----------
        z : int | None = None
            直径を取得するzスライスを指定
        auto_z : bool = True
            Trueの場合、粒子が最初に現れるスライスの次のスライスを対象とする
        header : bool = True
            CSVにヘッダーを付けるかどうか
        """
        for analyzer in self.particle_analyzers:
            analyzer.output_diameter_csv(z, auto_z, header)
