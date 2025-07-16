from modules import Config, ParticleAnalyzer, ImageInterface


"""
メインプロセスの記述する箇所
微調整するときはここか modules/config.py を編集する。
"""

def main():
    """メイン処理"""
    paths = [
        "path/to/binary_image_stack.tif",
    ]
    for path in paths:
        config = Config()
        image_interface = ImageInterface(path)
        analyzer = ParticleAnalyzer(config, image_interface)
        analyzer.run_analysis()
        analyzer.output_particle_image()
        analyzer.plot_diameter_histogram(title="", auto_z=False, density=False, xlim=(0,25))

if __name__ == "__main__":
    main()