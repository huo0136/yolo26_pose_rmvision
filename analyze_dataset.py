from __future__ import annotations

# 命令行参数解析（类似 C++ 里的命令行参数处理）
import argparse
# 随机数（用于打乱验证集样本）
import random
# 计数器（统计每类样本数量）
from collections import Counter
# 路径操作（类似 C++ std::filesystem::path）
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 不弹窗，直接把图保存到文件（服务器无显示器时需要）
import matplotlib.pyplot as plt  # 绘图库（类似 MATLAB / gnuplot）
import numpy as np  # 数值计算数组（类似 C++ Eigen / std::valarray）
import yaml  # 解析 YAML 配置文件
from PIL import Image, ImageDraw  # 图片读取与绘图（Python 图像库 Pillow）

# 当前脚本所在目录
BASE_DIR = Path(__file__).resolve().parent
# 默认数据集配置文件路径
DEFAULT_DATA_YAML = BASE_DIR / "armor_pose.yaml"
# 支持的图片后缀
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 字母箱缩放后的目标短边尺寸（像训练时的 imgsz）
SIZES = (640, 960)


def resolve_data(data_yaml: Path) -> tuple[dict, Path, list[str]]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    configured_root = Path(data.get("path", data_yaml.parent))
    root = configured_root if configured_root.is_absolute() else data_yaml.parent / configured_root
    raw_names = data["names"]
    if isinstance(raw_names, list):
        names = [str(value) for value in raw_names]
    else:
        names = [str(raw_names[index]) for index in range(len(raw_names))]
    return data, root.resolve(), names


def load_samples(data: dict, root: Path, names: list[str]) -> tuple[dict[str, Counter], dict[int, list[float]], list[tuple[Path, list[tuple[int, float, float, float, float, list[tuple[float, float]]]]]]]:
    class_counts = {split: Counter() for split in ("train", "val")}
    short_side_by_size = {size: [] for size in SIZES}
    validation_samples: list[tuple[Path, list[tuple[int, float, float, float, float, list[tuple[float, float]]]]]] = []

    for split in ("train", "val"):
        image_dir = Path(data[split])
        if not image_dir.is_absolute():
            image_dir = root / image_dir
        label_dir = image_dir.parent / "labels"
        image_paths = sorted(
            (path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda path: str(path),
        )
        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception:
                continue

            annotations: list[tuple[int, float, float, float, float, list[tuple[float, float]]]] = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if not values:
                    continue
                if len(values) != 13:
                    continue
                row = [float(value) for value in values]
                class_id = int(row[0])
                if not 0 <= class_id < len(names):
                    continue
                cx, cy, box_w, box_h = row[1:5]
                points = [(row[i], row[i + 1]) for i in range(5, 13, 2)]
                annotations.append((class_id, cx, cy, box_w, box_h, points))
                class_counts[split][class_id] += 1
                for size in SIZES:
                    scale = min(size / width, size / height)
                    short_side_by_size[size].append(min(box_w * width, box_h * height) * scale)

            if split == "val" and annotations:
                validation_samples.append((image_path, annotations))
    return class_counts, short_side_by_size, validation_samples


def plot_class_counts(class_counts: dict[str, Counter], names: list[str], output: Path) -> None:
    x = np.arange(len(names))
    train = [class_counts["train"].get(index, 0) for index in range(len(names))]
    val = [class_counts["val"].get(index, 0) for index in range(len(names))]
    width = 0.39

    fig, ax = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    ax.bar(x - width / 2, train, width, label="Train", color="#2864a5")
    ax.bar(x + width / 2, val, width, label="Validation", color="#ef8a32")
    ax.set_xticks(x, names)
    ax.set_ylabel("Annotated armor instances")
    ax.set_title("Class distribution by split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_target_sizes(values: dict[int, list[float]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6), constrained_layout=True)
    bins = np.linspace(0, 180, 46)
    for size, color in ((640, "#2864a5"), (960, "#d25b45")):
        data = np.asarray(values[size], dtype=np.float32)
        ax.hist(data, bins=bins, histtype="step", linewidth=2.3, label=f"imgsz {size}", color=color)
        if data.size:
            median = float(np.median(data))
            ax.axvline(median, color=color, linestyle="--", alpha=0.7)
            print(
                f"imgsz {size}: target short-side median={median:.1f}px, "
                f"below 10px={(data < 10).mean() * 100:.1f}%, "
                f"below 16px={(data < 16).mean() * 100:.1f}%"
            )
    ax.axvline(10, color="#777777", linestyle=":", label="10px reference")
    ax.axvline(16, color="#999999", linestyle="-.", label="16px reference")
    ax.set_xlabel("Shorter side of the bounding box (pixels after letterbox resize)")
    ax.set_ylabel("Instances")
    ax.set_title("Armor target size distribution")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_annotation_preview(
    samples: list[tuple[Path, list[tuple[int, float, float, float, float, list[tuple[float, float]]]]]],
    names: list[str],
    output: Path,
    limit: int,
) -> None:
    if not samples:
        print("没有可用的验证集标注图片，跳过预览图。")
        return
    random.Random(42).shuffle(samples)
    selected = samples[:limit]
    tile_w, tile_h = 480, 320
    canvas = Image.new("RGB", (tile_w * 2, tile_h * ((len(selected) + 1) // 2)), "white")
    draw_canvas = ImageDraw.Draw(canvas)

    for index, (image_path, annotations) in enumerate(selected):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        image.thumbnail((tile_w - 12, tile_h - 42))
        x0 = (index % 2) * tile_w + (tile_w - image.width) // 2
        y0 = (index // 2) * tile_h + 28
        canvas.paste(image, (x0, y0))
        draw_canvas.text(((index % 2) * tile_w + 8, (index // 2) * tile_h + 7), image_path.name, fill="black")
        draw_image = ImageDraw.Draw(canvas)
        original_w, original_h = Image.open(image_path).size
        scale_x = image.width / original_w
        scale_y = image.height / original_h
        for class_id, cx, cy, box_w, box_h, points in annotations:
            left = x0 + (cx - box_w / 2) * original_w * scale_x
            right = x0 + (cx + box_w / 2) * original_w * scale_x
            top = y0 + (cy - box_h / 2) * original_h * scale_y
            bottom = y0 + (cy + box_h / 2) * original_h * scale_y
            color = "#22d3ee"
            draw_image.rectangle((left, top, right, bottom), outline=color, width=2)
            draw_image.text((left, max(0, top - 14)), names[class_id], fill=color)
            for point_index, (px, py) in enumerate(points, 1):
                point_x = x0 + px * original_w * scale_x
                point_y = y0 + py * original_h * scale_y
                radius = 3
                draw_image.ellipse(
                    (point_x - radius, point_y - radius, point_x + radius, point_y + radius),
                    fill="#f43f5e",
                )
                draw_image.text((point_x + 3, point_y + 2), str(point_index), fill="#f43f5e")
    canvas.save(output, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成数据分布与标注预览图。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    data, root, names = resolve_data(args.data.resolve())
    class_counts, size_values, samples = load_samples(data, root, names)
    output_dir = BASE_DIR / "outputs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_class_counts(class_counts, names, output_dir / "class_distribution.png")
    plot_target_sizes(size_values, output_dir / "target_size_distribution.png")
    save_annotation_preview(samples, names, output_dir / "annotation_preview.jpg", args.samples)

    for split in ("train", "val"):
        total = sum(class_counts[split].values())
        print(f"{split}: {total} 个有效目标实例，{len(class_counts[split])} 个类别有标注")
    print(f"分析图已写入：{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

