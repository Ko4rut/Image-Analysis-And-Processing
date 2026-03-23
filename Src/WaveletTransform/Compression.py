import os
from pathlib import Path

import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os

# SSIM là tùy chọn, nếu chưa cài thì code vẫn chạy
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# Utility functions
# =========================
def load_grayscale_image(image_path: str) -> np.ndarray:
    """
    Đọc ảnh grayscale và trả về dạng float32 trong khoảng [0, 255].
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    return image.astype(np.float32)


def save_image(image: np.ndarray, save_path: str) -> None:
    """
    Lưu ảnh float/uint8 về file.
    """
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image_uint8)


def compute_mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.mean((original - reconstructed) ** 2))


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray, max_val: float = 255.0) -> float:
    mse = compute_mse(original, reconstructed)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10((max_val ** 2) / mse))


def compute_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float | None:
    if not HAS_SKIMAGE:
        return None
    return float(
        ssim(
            original.astype(np.uint8),
            reconstructed.astype(np.uint8),
            data_range=255
        )
    )


# =========================
# Wavelet compression
# =========================
def compress_wavelet_topk(
    image: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
    keep_ratio: float = 0.10,
    mode: str = "periodization",
) -> dict:
    """
    Nén ảnh bằng Wavelet theo cách giữ lại top-k hệ số lớn nhất.

    Parameters
    ----------
    image : np.ndarray
        Ảnh grayscale float32 [0, 255]
    wavelet : str
        Loại wavelet, ví dụ: 'haar', 'db2', 'db4', 'sym4', ...
    level : int
        Số mức phân rã
    keep_ratio : float
        Tỉ lệ hệ số giữ lại, ví dụ 0.1 = giữ 10%
    mode : str
        Chế độ biên của pywt

    Returns
    -------
    dict gồm:
        reconstructed, threshold, kept_coeffs, total_coeffs,
        actual_keep_ratio, approx_compression_ratio, mse, psnr, ssim
    """
    if not (0 < keep_ratio <= 1):
        raise ValueError("keep_ratio phải nằm trong khoảng (0, 1].")

    # Phân rã wavelet 2D
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level, mode=mode)

    # Chuyển toàn bộ hệ số về 1 mảng để dễ threshold
    coeff_array, coeff_slices = pywt.coeffs_to_array(coeffs)

    abs_coeffs = np.abs(coeff_array).ravel()
    total_coeffs = abs_coeffs.size

    # Số hệ số cần giữ lại
    k = max(1, int(np.floor(keep_ratio * total_coeffs)))

    # Tìm ngưỡng để giữ top-k hệ số lớn nhất
    threshold = np.partition(abs_coeffs, -k)[-k]

    # Tạo mask giữ lại hệ số đủ lớn
    mask = np.abs(coeff_array) >= threshold
    compressed_array = coeff_array * mask

    kept_coeffs = int(np.count_nonzero(compressed_array))
    actual_keep_ratio = kept_coeffs / total_coeffs
    approx_compression_ratio = total_coeffs / max(kept_coeffs, 1)

    # Chuyển ngược về cấu trúc coeffs
    compressed_coeffs = pywt.array_to_coeffs(
        compressed_array,
        coeff_slices,
        output_format="wavedec2"
    )

    # Tái tạo ảnh
    reconstructed = pywt.waverec2(compressed_coeffs, wavelet=wavelet, mode=mode)

    # waverec2 đôi khi ra hơi dư 1 chút do padding => crop lại
    reconstructed = reconstructed[: image.shape[0], : image.shape[1]]
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.float32)

    # Metrics
    mse_value = compute_mse(image, reconstructed)
    psnr_value = compute_psnr(image, reconstructed)
    ssim_value = compute_ssim(image, reconstructed)

    return {
        "reconstructed": reconstructed,
        "threshold": float(threshold),
        "kept_coeffs": kept_coeffs,
        "total_coeffs": total_coeffs,
        "actual_keep_ratio": float(actual_keep_ratio),
        "approx_compression_ratio": float(approx_compression_ratio),
        "mse": mse_value,
        "psnr": psnr_value,
        "ssim": ssim_value,
    }


# =========================
# Visualization
# =========================
def save_comparison_figure(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_path: str,
    title: str = "Wavelet Compression Result"
) -> None:
    """
    Lưu hình so sánh ảnh gốc / ảnh tái tạo / sai khác tuyệt đối.
    """
    diff = np.abs(original - reconstructed)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed, cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap="gray")
    plt.title("Absolute Difference")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# Main experiment
# =========================
def run_experiment():
    BASE_DIR = Path(__file__).resolve().parent
    image_path = os.path.join(BASE_DIR, "..", "..", "Images", "lena_std.tif")
    output_dir = BASE_DIR / "output_wavelet"

    image = load_grayscale_image(str(image_path))

    wavelet_name = "db4"
    level = 3

    # Các mức giữ hệ số để thử nghiệm
    keep_ratios = [0.20, 0.10, 0.05, 0.01]

    print("=" * 80)
    print("WAVELET IMAGE COMPRESSION EXPERIMENT")
    print("=" * 80)
    print(f"Image       : {image_path}")
    print(f"Wavelet     : {wavelet_name}")
    print(f"Level       : {level}")
    print(f"Image shape : {image.shape}")
    print("-" * 80)

    results = []

    for keep_ratio in keep_ratios:
        result = compress_wavelet_topk(
            image=image,
            wavelet=wavelet_name,
            level=level,
            keep_ratio=keep_ratio,
            mode="periodization",
        )

        reconstructed = result["reconstructed"]

        # Tên file output
        tag = f"keep_{int(keep_ratio * 100):02d}"
        recon_path = output_dir / f"reconstructed_{tag}.png"
        fig_path = output_dir / f"comparison_{tag}.png"

        save_image(reconstructed, str(recon_path))
        save_comparison_figure(
            original=image,
            reconstructed=reconstructed,
            save_path=str(fig_path),
            title=f"Wavelet={wavelet_name}, level={level}, keep={keep_ratio:.0%}"
        )

        results.append({
            "keep_ratio_target": keep_ratio,
            "keep_ratio_actual": result["actual_keep_ratio"],
            "approx_CR": result["approx_compression_ratio"],
            "MSE": result["mse"],
            "PSNR": result["psnr"],
            "SSIM": result["ssim"],
        })

    # In bảng kết quả
    print(f"{'Target keep':>12} | {'Actual keep':>12} | {'Approx CR':>10} | {'MSE':>12} | {'PSNR':>10} | {'SSIM':>10}")
    print("-" * 80)
    for r in results:
        ssim_text = f"{r['SSIM']:.6f}" if r["SSIM"] is not None else "N/A"
    print(
        f"{r['keep_ratio_target']:>12.2%} | "
        f"{r['keep_ratio_actual']:>12.2%} | "
        f"{r['approx_CR']:>10.4f} | "
        f"{r['MSE']:>12.4f} | "
        f"{r['PSNR']:>10.4f} | "
        f"{ssim_text:>10}"
    )

    print("-" * 80)
    print(f"Kết quả đã lưu tại: {output_dir}")



import os
import cv2 as cv
import matplotlib.pyplot as plt


def show_8_images(image_paths, titles=None, window_title="8 Images"):
    """
    Hiển thị 8 ảnh trên màn hình theo dạng 2 hàng x 4 cột.

    Parameters
    ----------
    image_paths : list[str]
        Danh sách đúng 8 đường dẫn ảnh.
    titles : list[str] | None
        Tiêu đề từng ảnh. Nếu None thì lấy tên file.
    window_title : str
        Tiêu đề chung của figure.
    """
    if len(image_paths) != 8:
        raise ValueError("Phải truyền đúng 8 đường dẫn ảnh.")

    if titles is None:
        titles = [os.path.basename(p) for p in image_paths]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(window_title, fontsize=18)

    for ax, img_path, title in zip(axes.ravel(), image_paths, titles):
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")

        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

BASE_DIR = os.path.dirname(__file__)
output_dir = os.path.join(BASE_DIR, "output_wavelet")

image_paths = [
    os.path.join(output_dir, "comparison_keep_01.png"),
    os.path.join(output_dir, "comparison_keep_05.png"),
    os.path.join(output_dir, "comparison_keep_10.png"),
    os.path.join(output_dir, "comparison_keep_20.png"),
    os.path.join(output_dir, "reconstructed_keep_01.png"),
    os.path.join(output_dir, "reconstructed_keep_05.png"),
    os.path.join(output_dir, "reconstructed_keep_10.png"),
    os.path.join(output_dir, "reconstructed_keep_20.png"),
]

show_8_images(image_paths, window_title="Wavelet Output")


# if __name__ == "__main__":
#     run_experiment()