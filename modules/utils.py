import os
import glob


def decode_paths(paths: list[str]):
    """パスのリストを受け取り、ディレクトリ内のすべての.tifファイルのパスに展開する"""

    decoded_paths = []
    for path in paths:
        if os.path.isdir(path):
            parent_dir = os.path.basename(path)
            files = glob.glob(os.path.join(path, "*.tif"))
            decoded_paths.extend([(f, parent_dir) for f in files])
        elif os.path.isfile(path) and path.lower().endswith(".tif"):
            decoded_paths.append((path, None))
    return sorted(decoded_paths, key=lambda x: x[0])
