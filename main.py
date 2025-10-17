from modules import Config, ParticleAnalyzer, ImageInterface, BatchParticleAnalyzer, decode_paths


"""
メインプロセスの記述する箇所
微調整するときはここか modules/config.py を編集する。
"""


paths = [
    "path/to/binary_image_stack_01.tif",
    "path/to/binary_image_stack_02.tif",
    "path/to/binary_image_stack_03.tif",
]


def main():
    """メイン処理"""
    config = Config()
    analyzers = [ParticleAnalyzer(config, ImageInterface(path, parent)) for path, parent in decode_paths(paths)]
    batch_analyzer = BatchParticleAnalyzer(analyzers, config)

    batch_analyzer.run_analysis()
    batch_analyzer.output_particle_image()
    batch_analyzer.plot_diameter_histogram()
    batch_analyzer.output_diameter_csv()
    batch_analyzer.output_summary_csv()


if __name__ == "__main__":
    main()
