import math
from tqdm import tqdm

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

    def run_analysis(self):
        """登録されているすべてのParticleAnalyzerで解析を実行する"""
        with tqdm(
            total=len(self.particle_analyzers),
            desc="Analyzing",
            leave=True,
            bar_format=f"{{l_bar}}{{bar}} | {{n_fmt}}/{{total_fmt}}",
        ) as pbar:
            for analyzer in self.particle_analyzers:
                pbar.set_description(f"Analyzing '{analyzer.image_interface.filename}.tif'")
                analyzer.run_analysis()
                pbar.update(1)
            pbar.set_description(f"Completed")
            pbar.close()

    def output_summary_csv(self, ranges: list[tuple[int | None, int | None]]):
        """登録されているすべてのParticleAnalyzerでサマリーCSVを出力する

        Parameters
        ----------
        ranges : list[tuple[int | None, int | None]]
            集計対象とする直径の範囲を指定するタプルのリスト
        """
        for analyzer in self.particle_analyzers:
            analyzer.output_summary_csv(ranges)

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

    def output_diameter_csv(self, auto_z: bool = True, header: bool = True):
        """登録されているすべてのParticleAnalyzerで直径の生データCSVを出力する

        Parameters
        ----------
        auto_z : bool = True
            Trueの場合、粒子が最初に現れるスライスの次のスライスを対象とする
        header : bool = True
            CSVにヘッダーを付けるかどうか
        """
        for analyzer in self.particle_analyzers:
            analyzer.output_diameter_csv(auto_z, header)
