import math
from tqdm import tqdm

from modules.particle_analyzer import ParticleAnalyzer, Config


class BatchParticleAnalyzer:
    def __init__(self, particle_analyzers: list[ParticleAnalyzer], config: Config):
        self.particle_analyzers = particle_analyzers
        self.config = config

    def run_analysis(self):
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
        for analyzer in self.particle_analyzers:
            analyzer.output_summary_csv(ranges)

    def output_particle_image(self):
        for analyzer in self.particle_analyzers:
            analyzer.output_particle_image()

    def get_max_diameter(self, z: int | None = None, auto_z: bool = False) -> float:
        return max(
            analyzer.get_max_diameter(z, auto_z) for analyzer in self.particle_analyzers
        )

    def plot_diameter_histogram(
        self,
        add_title: bool = True,
        z: float | None = None,
        auto_z: bool = False,
        density: bool = False,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        bin_width: float | None = None,
    ):
        limit_diameter = self.get_max_diameter(z, auto_z) if not xlim else xlim[1]
        step = self.config.DEFAULT_HISTOGRAM_BIN_WIDTH if not bin_width else bin_width
        bins = math.ceil(limit_diameter / step)
        x_range = (0, int(step * bins) + step) if not xlim else xlim

        # ylimを各画像の最大値に合わせる場合、全画像の最大値を取得
        y_limit = (
            max(
                [
                    analyzer.calc_mode_bin(step, density, z, auto_z)[1]
                    for analyzer in self.particle_analyzers
                ]
            )
            if ylim is None
            else ylim[1]
        )
        y_range = (0, y_limit * 1.1) if ylim is None else ylim

        for analyzer in self.particle_analyzers:
            title = analyzer.image_interface.filename if add_title else ""
            analyzer.plot_diameter_histogram(
                title, z, auto_z, density, x_range, y_range, bin_width
            )

    def output_diameter_csv(self, auto_z: bool = True, header: bool = True):
        for analyzer in self.particle_analyzers:
            analyzer.output_diameter_csv(auto_z, header)
