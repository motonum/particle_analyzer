import cv2
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, mark_boundaries
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import threshold_otsu
from skimage.util import img_as_ubyte
from sklearn.neighbors import NearestNeighbors


from modules.particle import Particle
from modules.config import Config
from modules.image_interface import ImageInterface
from modules.circle import Circle, IdentifiedCircle


class ParticleAnalyzer:
    """多層画像から粒子を解析するクラス

    マルチページTIFF画像を保持し粒子情報を抽出する

    """

    def __init__(self, config: Config, image_interface: ImageInterface):
        self.config = config
        self.image_interface: ImageInterface = image_interface
        self.image_stack = self._load_images(image_interface.img_path)
        self.segmented_stack = []
        self.circles_by_slice: list[list[Circle]] = []
        self.particle_repository: dict[int, Particle] = {}
        self.next_particle_id = 0
        self.identified_circles_by_slice: list[list[IdentifiedCircle]] = []

    def _load_images(self, path: str):
        """複数ページのTIF画像を読み込む

        Parameters
        ----------
        path: string
            マルチページTIFFファイルの場所を表す文字列
            読み込まれる画像は二値化した画像のスタック

        Returns
        -------
        images: Sequence[MatLike]
            読み込んだ画像の配列
        """
        _, images = cv2.imreadmulti(filename=path, mats=[], flags=cv2.IMREAD_GRAYSCALE)
        return images

    def _segment_slice(self, img):
        """単一の画像をセグメンテーションする

        穴埋め、ノイズ除去、Watershedを実行し、粒子ごとに切り分ける
        Watershedのシードは距離変換をしたときの極大値を用いている

        Parameters
        ----------
        img: MatLike
            画像データ(2次元配列)

        Returns
        -------
        images: MatLike
            セグメンテーション後の画像
        """

        # 前処理
        # blurred = cv2.GaussianBlur(img, self.config.GAUSSIAN_BLUR_KERNEL, 0)
        binary = img_as_ubyte(img > threshold_otsu(img))

        # ノイズ除去
        fill_holes_structure = np.ones(self.config.FILL_HOLES_STRUCTURE_SIZE)
        filled = ndi.binary_fill_holes(binary, structure=fill_holes_structure)

        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.config.MORPHOLOGY_KERNEL_SIZE)
        closed = cv2.morphologyEx(
            img_as_ubyte(filled),
            cv2.MORPH_CLOSE,
            morph_kernel,
            iterations=self.config.MORPHOLOGY_ITERATIONS,
        )
        opened = cv2.morphologyEx(
            closed,
            cv2.MORPH_OPEN,
            morph_kernel,
            iterations=self.config.MORPHOLOGY_ITERATIONS,
        )

        # Watershedアルゴリズム
        dist_transform = ndi.distance_transform_edt(opened)
        min_dist = self.config.DOTS_PER_MICRON * self.config.PEAK_MIN_DISTANCE_RATIO
        thresh_abs = self.config.DOTS_PER_MICRON * self.config.PEAK_THRESHOLD_RATIO

        local_max_coords = peak_local_max(
            dist_transform,
            min_distance=int(min_dist),
            labels=opened,
            threshold_abs=thresh_abs,
        )

        sorted_coords = sorted(
            local_max_coords,
            key=lambda pos: dist_transform[pos[0]][pos[1]],
            reverse=True,
        )
        canvas = np.zeros(dist_transform.shape, np.uint8)

        merged_peaks = []
        for y, x in sorted_coords:
            dist = dist_transform[y][x]
            if canvas[y][x] != 255:
                cv2.circle(canvas, (x, y), int(dist), 255, cv2.FILLED)
                merged_peaks.append([y, x])

        merged_peaks = np.asarray(merged_peaks)

        local_max_mask = np.zeros(dist_transform.shape, dtype=bool)
        local_max_mask[tuple(merged_peaks.T)] = True

        markers, _ = ndi.label(local_max_mask)
        labels = watershed(-dist_transform, markers, mask=opened)

        # 結果を二値画像として返す
        segmented_img = mark_boundaries(gray2rgb(opened), labels, color=(0, 0, 0), mode="thin")
        segmented_gray = rgb2gray(segmented_img)
        return img_as_ubyte(segmented_gray > threshold_otsu(segmented_gray))

    def _fit_circles(self, binary_img):
        """二値画像から円を検出し、フィルタリングする

        セグメンテーションされた粒子像に対し輪郭検出し円でフィッティングし、
        検出された円の中心座標と半径をリストを返す

        Parameters
        ----------
        binary_img: MatLike
            セグメンテーション後の単一画像

        Returns
        -------
        detected_circles: list[Circle]
            検出された円の中心座標と半径のリスト
        """

        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_circles: list[Circle] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area == 0:
                continue

            (x, y), _ = cv2.minEnclosingCircle(cnt)

            radius = np.sqrt(area / np.pi)

            # フィルタリング
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter**2))

            min_radius = self.config.MIN_DIAMETER_MICRON / 2 * self.config.DOTS_PER_MICRON
            if radius < min_radius or circularity < self.config.MIN_CIRCULARITY:
                continue

            detected_circles.append(Circle((x, y), radius))

        return detected_circles

    def _generate_random_color(self) -> tuple[np.uint8, np.uint8, np.uint8]:
        """ランダムな色のtuple (r,g,b) を生成する関数"""
        return tuple(np.random.randint(0, 256, 3).tolist())

    def _create_new_particle(self, slice_index: int, circle: Circle):
        particle_id = self.next_particle_id
        color = self._generate_random_color()
        particle = Particle(
            particle_id,
            slice_index,
            circle.radius,
            circle.coord,
            color,
            self.config.DOTS_PER_MICRON,
        )
        self.particle_repository[particle_id] = particle
        self.next_particle_id += 1
        return particle

    def _track_particles(self, strict: bool = True):
        """スライス間で粒子を追跡する

        連続する2枚のスライス像を見て、中心が相互最近傍の点であり、半径が大きい方の円の中心座標が、
        半径が小さい方の円の内部に収まっていれば、同一の粒子由来の像であるとして粒子を追跡する関数。
        この関数でしていることは大きく、

        1. 同一の粒子由来のものには同じID(シリアルナンバー)を振り、particle_repositoryに登録すること、
        2. self._fit_circlesの返り値の円にID情報を付与したリストを作成すること

        の2つ。

        Parameters
        ----------
        strict: bool = True
            Trueのとき、半径が小さい方の円の内部に収まっていれば、同一の粒子由来の像であるとして粒子を追跡する
            Falseのときはどちらかかの円の中心がもう一方の円の内部に収まっていれば同一粒子とみなす

        Returns
        -------
        detected_circles: list[list[IdentifiedCircle]]
            IDを付与したスライスごとの円のリスト
        """
        # 最初のスライスを初期化
        initial_circles: list[IdentifiedCircle] = []
        for circle in self.circles_by_slice[0]:
            particle = self._create_new_particle(0, circle)
            initial_circles.append(IdentifiedCircle(circle.coord, circle.radius, particle.id))

        identified_circles_by_slice = [initial_circles]

        # 2枚目以降のスライスを処理
        for i in range(1, len(self.circles_by_slice)):
            prev_slice_index = i - 1
            current_slice_circles = self.circles_by_slice[i]
            prev_slice_identified_circles = identified_circles_by_slice[prev_slice_index]

            newly_identified_circles = []

            if not prev_slice_identified_circles:
                # 前のスライスに粒子がない場合、すべて新しい粒子として登録
                for circle in current_slice_circles:
                    particle = self._create_new_particle(i, circle)
                    newly_identified_circles.append(IdentifiedCircle(circle.coord, circle.radius, particle.id))
            else:
                prev_coords = np.array([c.coord for c in prev_slice_identified_circles])
                nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(prev_coords)

                for circle in current_slice_circles:
                    dist, index = nn.kneighbors([circle.coord])

                    # 最近傍の粒子情報を取得
                    closest_particle_index: int = index[0][0]
                    prev_particle_info = prev_slice_identified_circles[closest_particle_index]

                    # 相互最近傍であるか、かつ距離が妥当かをチェック
                    # (この実装は簡略化のため、一方向の最近傍のみチェック)
                    # 距離が両方の半径の小さい方より近ければ同一粒子とみなす
                    min_radius_threshold = (
                        min(circle.radius, prev_particle_info.radius)
                        if strict
                        else max(circle.radius, prev_particle_info.radius)
                    )
                    if dist[0][0] < min_radius_threshold:
                        particle_id = prev_particle_info.id
                        self.particle_repository[particle_id].add_slice(i, circle.radius, circle.coord)
                        newly_identified_circles.append(IdentifiedCircle(circle.coord, circle.radius, particle_id))
                    else:
                        # 新しい粒子として登録
                        particle = self._create_new_particle(i, circle)
                        newly_identified_circles.append(IdentifiedCircle(circle.coord, circle.radius, particle.id))

            identified_circles_by_slice.append(newly_identified_circles)

        return identified_circles_by_slice

    def _get_diameters(self, single_z: bool = False, z: int | None = None):
        """粒子の直径のリストを作成する

        Parameters
        ----------
        single_z: bool = False
            Trueのときは、単一スライスに存在する粒子のみをカウントする
        z: int | None = None
            指定したzスライスに存在する粒子をカウントする(zはone-based)

        Returns
        -------
        diameters: list[float]
            粒子直径のリスト
        target_z: int
            調査対象のzスライス（zはone-indexed）
            全粒子が対象のときは-1
        """
        diameters: list[float] = []

        if z:
            z -= 1

        if single_z and z is None:
            for i, plane in enumerate(self.identified_circles_by_slice):
                if len(plane):
                    if i + 1 in [k for k, v in enumerate(self.identified_circles_by_slice)]:
                        diameters = [
                            self.particle_repository[c.id].diameter_micron
                            for c in self.identified_circles_by_slice[i + 1]
                        ]
                        z = i + 1
                    else:
                        diameters = [self.particle_repository[c.id].diameter_micron for c in plane]
                        z = i
                    break
        elif z in [k for k, v in enumerate(self.identified_circles_by_slice)]:
            diameters = [self.particle_repository[c.id].diameter_micron for c in self.identified_circles_by_slice[z]]
        else:
            diameters = [p.diameter_micron for p in self.particle_repository.values()]

        target_z = None if z is None else z + 1

        return diameters, target_z

    def get_max_diameter(self, single_z: bool = False, z: int | None = None) -> float:
        """粒子の最大直径を取得する

        Parameters
        ----------
        single_z: bool = False
            Trueのときは、単一スライスに存在する粒子のみをカウントする
        z: int | None = None
            指定したzスライスに存在する粒子をカウントする(zはone-based)

        Returns
        -------
        max_diameter: float
            粒子の最大直径(μm)
        """
        diameters, _ = self._get_diameters(single_z, z)
        if not diameters:
            return 0.0
        return max(diameters)

    def calc_mode_bin(
        self,
        bin_width: float | None = None,
        density: bool = False,
        single_z: bool = False,
        z: int | None = None,
        upper_limit: float | None = None,
    ) -> tuple[float, float]:
        """粒子の最頻値直径を取得する

        Parameters
        ----------
        bin_width: float = None
            ヒストグラムのビン幅(μm)
        density: bool = False
            Trueのとき、ヒストグラムの縦軸を相対度数とする
        single_z: bool = False
            Trueのときは、単一スライスに存在する粒子のみをカウントする
        z: int | None = None
            指定したzスライスに存在する粒子をカウントする(zはone-based)

        Returns
        -------
        mode_diameter: float
            粒子の最頻値直径(μm)
        quantity: float
            最頻値の出現数(個)または相対度数
        """
        diameters, _ = self._get_diameters(single_z, z)
        step = self.config.DEFAULT_HISTOGRAM_BIN_WIDTH if not bin_width else bin_width

        if upper_limit is not None:
            # 指定された制限値を超える値は全て制限値として扱う(最後のビンにまとめる)
            diameters = [min(d, upper_limit) for d in diameters]
            bins = math.ceil(upper_limit / step) + 1
        else:
            bins = math.ceil(max(diameters) / step) + 1

        range = (0.0, int(step * bins))
        hist, _ = np.histogram(diameters, bins=bins, density=density, range=range)
        mode_index = np.argmax(hist)
        mode_diameter = (mode_index * step + (mode_index + 1) * step) / 2
        quantity = hist[mode_index]
        return mode_diameter, quantity

    def run_analysis(self, strict_tracking: bool = True):
        """解析を実行する"""
        # 1. 各スライスの粒子を検出
        for img in tqdm(
            self.image_stack,
            desc="Analyzing slices",
            leave=False,
            bar_format=f"{{l_bar}}{{bar}} | {{n_fmt}}/{{total_fmt}}",
        ):
            binary_img = self._segment_slice(img)
            self.segmented_stack.append(binary_img)
            circles = self._fit_circles(binary_img)
            self.circles_by_slice.append(circles)

        # 2. スライス間で粒子を追跡
        self.identified_circles_by_slice = self._track_particles(strict=strict_tracking)

    def output_particle_image(self):
        """セグメンテーション後の画像と検出された粒子を円でフィッティングしたものを色付けして画像として出力する"""
        output_stack = []
        height, width = self.image_stack[0].shape
        for slice_circles in self.identified_circles_by_slice:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            for c_info in slice_circles:
                particle = self.particle_repository[c_info.id]
                coord_int = tuple(int(pos) for pos in c_info.coord)
                cv2.circle(canvas, coord_int, int(c_info.radius), particle.color, cv2.FILLED)
            output_stack.append(canvas)

        if not os.path.exists(self.image_interface.output_detected_dir):
            os.makedirs(self.image_interface.output_detected_dir)

        cv2.imwritemulti(self.image_interface.output_detected_path, output_stack)
        print(f"Detected particles image saved to {self.image_interface.output_detected_path}")

        if not os.path.exists(self.image_interface.output_segmented_dir):
            os.makedirs(self.image_interface.output_segmented_dir)

        cv2.imwritemulti(self.image_interface.output_segmented_path, self.segmented_stack)
        print(f"Segmented particles image saved to {self.image_interface.output_segmented_path}")

    def plot_diameter_histogram(
        self,
        title="",
        single_z: bool = False,
        z: int | None = None,
        density: bool = True,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        bin_width: float | None = None,
        upper_limit: float | None = None,
    ):
        """粒子の直径のヒストグラムをプロットする

        Parameters
        ----------
        title: str = ""
            ヒストグラムの上部に表示するタイトル
        single_z: bool = False
            Trueのときは、単一スライスに存在する粒子のみをカウントする
        z: int | None = None
            カウント対象となる粒子の存在zスライスの位置(zはone-based)
        density: bool = False
            Trueのとき、ヒストグラムの縦軸を相対度数とする
        xlim: tuple[float, float]
            グラフの横軸の表示範囲
        ylim: tuple[float, float]
            グラフの縦軸の表示範囲
        bin_width: float | None = None
            ヒストグラムのビン幅(μm)

        Returns
        -------
        None
        """
        diameters, target_z = self._get_diameters(single_z, z)

        if not diameters:
            print("\033[31mNo particles found to plot histogram.\033[0m")
            return
        plt.rcParams["font.size"] = 15

        step = self.config.DEFAULT_HISTOGRAM_BIN_WIDTH if not bin_width else bin_width

        if upper_limit is not None:
            diameters = [min(d, upper_limit) for d in diameters]
            # upper_limitが設定されている場合、最大値はupper_limitになる想定だが
            # ビン計算のためにmax_diameterを設定
            max_diameter = upper_limit
        else:
            max_diameter = max(diameters) if not xlim else xlim[1]

        bins = math.ceil(max_diameter / step) + 1
        x_range = (0, int(step * bins)) if not xlim else xlim
        plt.figure(figsize=(5, 5))
        _, _, patches = plt.hist(diameters, range=x_range, bins=bins, density=density)

        if upper_limit is not None:
            # 最後のビンの色を変える
            if patches:
                patches[-1].set_facecolor("red")

            tick_step = upper_limit // 5
            ticks = np.arange(0, upper_limit + 1, tick_step)

            if ticks[-1] != upper_limit:
                ticks = np.append(ticks, upper_limit)

            plt.xticks(ticks)

            # ラベルの修正
            current_locs, current_labels = plt.xticks()
            new_labels = []
            for loc in current_locs:
                if math.isclose(loc, upper_limit):
                    new_labels.append(f"≥{int(loc)}")
                else:
                    new_labels.append(f"{int(loc)}")
            plt.xticks(current_locs, new_labels)

        if title:
            plt.title(title, fontsize=18)
        plt.xlabel("Diameter (μm)", fontsize=18)
        plt.ylabel(f"{'Relative' if density else 'Absolute'} Frequency", fontsize=18)
        if ylim:
            plt.ylim(ylim)
        if not os.path.exists(self.image_interface.output_histogram_dir):
            os.makedirs(self.image_interface.output_histogram_dir)
        plt.tight_layout()
        plt.savefig(self.image_interface.output_histogram_path(target_z))
        print(f"Histogram saved to {self.image_interface.output_histogram_path(target_z)}")
        plt.close()

    def output_diameter_csv(
        self,
        single_z: bool = False,
        z: int | None = None,
        header: bool = True,
    ):
        """検出した粒子径のCSVファイルを出力する関数

        与えられたzとsingle_zに対応する粒子径のCSVを出力する。
        zとsingle_zがいずれもデフォルト値のとき、zを限定せず全範囲の直径を出力する。

        Parameters
        ----------
        single_z: bool = False
            Trueのときは、単一スライスに存在する粒子のみをカウントする
        z: int | None = None
            カウント対象となる粒子の存在zスライスの位置(zはone-based)
        header: bool = True
            CSVファイルのヘッダの表示をするかしないか

        Returns
        -------
        None
        """
        diameters, target_z = self._get_diameters(single_z, z)

        if not os.path.exists(self.image_interface.output_csv_dir):
            os.makedirs(self.image_interface.output_csv_dir)

        with open(self.image_interface.output_csv_path(target_z), "w") as f:
            writer = csv.writer(f)
            header and writer.writerow(["No.", "diameter[um]"])
            writer.writerows([[i + 1, d] for i, d in enumerate(diameters)])

        print(f"CSV saved to {self.image_interface.output_csv_path(target_z)}")

    def output_summary_csv(
        self,
        research_ranges: list[tuple[float | None, float | None]] | None = None,
        single_z: bool = False,
        z: int | None = None,
    ):
        """粒子解析のサマリをCSVファイルとして出力する関数

        直径の平均、標準偏差、最大値、最小値、中央値、全粒子数を出力する。
        また、research_rangesに範囲を指定した場合、その範囲に含まれる粒子の割合(%)も出力する。

        Parameters
        ----------
        research_ranges: list[tuple[float | None, float | None]] | None = None
            粒子径の範囲を指定するタプルのリスト。Noneを指定すると上限・下限なしを意味する。
            例: [(None,5), (5,10), (10,None)]
        single_z: bool = False
            Trueのときは、単一スライスに存在する粒子のみをカウントする
        z: int | None = None
            カウント対象となる粒子の存在zスライスの位置(zはone-based)

        Returns
        -------
        None
        """
        diameters, target_z = self._get_diameters(single_z, z)

        if not os.path.exists(self.image_interface.output_summary_dir):
            os.makedirs(self.image_interface.output_summary_dir)

        with open(self.image_interface.output_summary_path(target_z), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Filename", self.image_interface.filename])
            writer.writerow(["Total Particles", len(diameters)])
            writer.writerow(["Mean Diameter [μm]", np.average(diameters)])
            writer.writerow(["Standard Deviation [μm]", np.std(diameters)])
            writer.writerow(["Median Diameter [μm]", np.median(diameters)])
            writer.writerow(["Min Diameter [μm]", min(diameters)])
            writer.writerow(["Max Diameter [μm]", max(diameters)])
            if research_ranges is not None:
                for r in research_ranges:

                    if r == (None, None):
                        continue
                    elif r[0] is None:
                        writer.writerow(
                            [
                                f"(,{r[1]}]",
                                len([d for d in diameters if d <= r[1]]) / len(diameters) * 100,
                            ]
                        )
                    elif r[1] is None:
                        writer.writerow(
                            [
                                f"[{r[0]},)",
                                len([d for d in diameters if r[0] <= d]) / len(diameters) * 100,
                            ]
                        )
                    elif r[0] > r[1]:
                        writer.writerow([f"[{r[1]},{r[0]}]", "Error: Invalid range"])
                    else:
                        writer.writerow(
                            [
                                f"[{r[0]},{r[1]}]",
                                len([d for d in diameters if r[0] <= d <= r[1]]) / len(diameters) * 100,
                            ]
                        )

        print(f"Summary CSV saved to {self.image_interface.output_summary_path(target_z)}")

    def print_summary(
        self,
        research_ranges: list[tuple[float | None, float | None]] | None = None,
        single_z: bool = False,
        z: int | None = None,
    ):
        """粒子解析のサマリをコンソールに出力する関数
        直径の平均、標準偏差、最大値、最小値、中央値、全粒子数を出力する。
        また、research_rangesに範囲を指定した場合、その範囲に含まれる粒子の割合(%)も出力する。

        Parameters
        ----------
        research_ranges: list[tuple[float | None, float | None]] | None = None
            粒子径の範囲を指定するタプルのリスト。Noneを指定すると上限・下限なしを意味する。
            例: [(None,5), (5,10), (10,None)]
        single_z: bool = False
            Trueのときは、単一スライスに存在する粒子のみをカウントする
        z: int | None = None
            カウント対象となる粒子の存在zスライスの位置(zはone-based)
        Returns
        -------
        None
        """

        diameters, _ = self._get_diameters(single_z, z)

        print("-------")
        print(self.image_interface.filename)
        print("全粒子数: ", len(diameters))
        print("平均: ", np.average(diameters))
        print("標準偏差: ", np.std(diameters))
        print("中央値: ", np.median(diameters))
        print("最小値: ", min(diameters))
        print("最大値: ", max(diameters))
        print("四分位範囲: ", np.percentile(diameters, 0.75) - np.percentile(diameters, 0.25))

        if research_ranges is not None:
            for r in research_ranges:
                if r == (None, None):
                    continue
                elif r[0] is None:
                    print(
                        f"(,{r[1]}]: ",
                        len([d for d in diameters if d <= r[1]]) / len(diameters) * 100,
                    )
                elif r[1] is None:
                    print(
                        f"[{r[0]},): ",
                        len([d for d in diameters if r[0] <= d]) / len(diameters) * 100,
                    )
                elif r[0] > r[1]:
                    print(f"[{r[1]},{r[0]}]: ", "\033[31mError: Invalid range\033[0m")
                else:
                    print(
                        f"[{r[0]},{r[1]}]: ",
                        len([d for d in diameters if r[0] <= d <= r[1]]) / len(diameters) * 100,
                    )
        print("-------")
