import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# =========================
# Utility functions
# =========================
def load_grayscale_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    return image.astype(np.float32)


def save_image(image: np.ndarray, save_path: str) -> None:
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
# Laplacian Pyramid
# =========================
def build_laplacian_pyramid(image: np.ndarray, levels: int = 3):
    """
    Trả về:
    - laplacian_bands: list các bandpass từ mức cao -> thấp
    - lowpass: residual lowpass cuối cùng
    """
    gaussian_pyramid = [image]
    laplacian_bands = []

    current = image
    for _ in range(levels):
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down)

        if up.shape != current.shape:
            up = cv2.resize(up, (current.shape[1], current.shape[0]))

        lap = current - up
        laplacian_bands.append(lap)
        gaussian_pyramid.append(down)
        current = down

    lowpass = gaussian_pyramid[-1]
    return laplacian_bands, lowpass


def reconstruct_laplacian_pyramid(laplacian_bands, lowpass):
    """
    Tái tạo từ lowpass và các bandpass.
    """
    current = lowpass.copy()

    for lap in reversed(laplacian_bands):
        up = cv2.pyrUp(current)
        if up.shape != lap.shape:
            up = cv2.resize(up, (lap.shape[1], lap.shape[0]))
        current = up + lap

    return current


# =========================
# Directional decomposition in frequency domain
# =========================
def fft2c(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(x))


def ifft2c(X: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifft2(np.fft.ifftshift(X)))


def make_directional_masks(shape, n_directions: int):
    """
    Tạo n_directions mask dạng wedge theo góc trong miền tần số.
    Tổng các mask xấp xỉ bằng 1.
    """
    h, w = shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    yy = y - cy
    xx = x - cx

    angle = np.arctan2(yy, xx)  # [-pi, pi]

    masks = []
    step = 2 * np.pi / n_directions

    for k in range(n_directions):
        center = -np.pi + (k + 0.5) * step

        # khoảng cách góc có wrap-around
        diff = np.angle(np.exp(1j * (angle - center)))
        mask = (np.abs(diff) <= step / 2).astype(np.float32)
        masks.append(mask)

    # chuẩn hóa để tổng mask = 1
    mask_sum = np.sum(masks, axis=0)
    mask_sum[mask_sum == 0] = 1.0
    masks = [m / mask_sum for m in masks]

    return masks


def directional_decompose(band: np.ndarray, n_directions: int):
    """
    Phân rã 1 bandpass thành nhiều hướng bằng mask góc trong miền Fourier.
    """
    F = fft2c(band)
    masks = make_directional_masks(band.shape, n_directions)

    directional_subbands = []
    for mask in masks:
        sub = ifft2c(F * mask)
        directional_subbands.append(sub.astype(np.float32))

    return directional_subbands


def directional_reconstruct(subbands):
    """
    Tổng các directional subbands để khôi phục bandpass.
    """
    return np.sum(subbands, axis=0).astype(np.float32)


# =========================
# Contourlet-style decomposition
# =========================
def contourlet_style_decompose(image: np.ndarray, levels: int = 3, directions_per_level=None):
    """
    directions_per_level ví dụ: [8, 4, 2]
    level đầu tiên là mức chi tiết cao nhất.
    """
    if directions_per_level is None:
        directions_per_level = [8, 4, 2]

    if len(directions_per_level) != levels:
        raise ValueError("Độ dài directions_per_level phải bằng levels.")

    laplacian_bands, lowpass = build_laplacian_pyramid(image, levels=levels)

    directional_pyramid = []
    for lap, ndir in zip(laplacian_bands, directions_per_level):
        subbands = directional_decompose(lap, ndir)
        directional_pyramid.append(subbands)

    return directional_pyramid, lowpass


def contourlet_style_reconstruct(directional_pyramid, lowpass):
    laplacian_bands = []
    for subbands in directional_pyramid:
        lap = directional_reconstruct(subbands)
        laplacian_bands.append(lap)

    reconstructed = reconstruct_laplacian_pyramid(laplacian_bands, lowpass)
    return np.clip(reconstructed, 0, 255).astype(np.float32)


# =========================
# Threshold top-k coefficients
# =========================
def compress_contourlet_topk(
    image: np.ndarray,
    levels: int = 3,
    directions_per_level=None,
    keep_ratio: float = 0.10
):
    if not (0 < keep_ratio <= 1):
        raise ValueError("keep_ratio phải nằm trong khoảng (0, 1].")

    directional_pyramid, lowpass = contourlet_style_decompose(
        image=image,
        levels=levels,
        directions_per_level=directions_per_level
    )

    # Gom toàn bộ hệ số directional + lowpass
    coeff_arrays = []
    meta = []

    # lowpass
    coeff_arrays.append(lowpass.ravel())
    meta.append(("lowpass", lowpass.shape))

    # directional subbands
    for i, level_subbands in enumerate(directional_pyramid):
        for j, sub in enumerate(level_subbands):
            coeff_arrays.append(sub.ravel())
            meta.append(("dir", i, j, sub.shape))

    all_coeffs = np.concatenate(coeff_arrays)
    abs_coeffs = np.abs(all_coeffs)

    total_coeffs = abs_coeffs.size
    k = max(1, int(np.floor(keep_ratio * total_coeffs)))
    threshold = np.partition(abs_coeffs, -k)[-k]

    # Apply threshold back
    compressed_directional_pyramid = []
    idx = 0

    # lowpass
    lowpass_size = lowpass.size
    lowpass_flat = all_coeffs[idx: idx + lowpass_size]
    lowpass_mask = np.abs(lowpass_flat) >= threshold
    compressed_lowpass = (lowpass_flat * lowpass_mask).reshape(lowpass.shape).astype(np.float32)
    idx += lowpass_size

    # directional
    for level_subbands in directional_pyramid:
        compressed_level = []
        for sub in level_subbands:
            sub_size = sub.size
            sub_flat = all_coeffs[idx: idx + sub_size]
            sub_mask = np.abs(sub_flat) >= threshold
            compressed_sub = (sub_flat * sub_mask).reshape(sub.shape).astype(np.float32)
            compressed_level.append(compressed_sub)
            idx += sub_size
        compressed_directional_pyramid.append(compressed_level)

    kept_coeffs = 0
    kept_coeffs += np.count_nonzero(compressed_lowpass)
    for level_subbands in compressed_directional_pyramid:
        for sub in level_subbands:
            kept_coeffs += np.count_nonzero(sub)

    reconstructed = contourlet_style_reconstruct(
        directional_pyramid=compressed_directional_pyramid,
        lowpass=compressed_lowpass
    )

    actual_keep_ratio = kept_coeffs / total_coeffs
    approx_compression_ratio = total_coeffs / max(kept_coeffs, 1)

    mse_value = compute_mse(image, reconstructed)
    psnr_value = compute_psnr(image, reconstructed)
    ssim_value = compute_ssim(image, reconstructed)

    return {
        "reconstructed": reconstructed,
        "threshold": float(threshold),
        "kept_coeffs": int(kept_coeffs),
        "total_coeffs": int(total_coeffs),
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
    title: str = "Contourlet-style Compression Result"
) -> None:
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
    image_path = BASE_DIR / ".." / ".." / "Images" / "lena_std.tif"
    output_dir = BASE_DIR / "output_contourlet"

    image = load_grayscale_image(str(image_path))

    levels = 3
    directions_per_level = [8, 4, 2]
    keep_ratios = [0.20, 0.10, 0.05, 0.01]

    print("=" * 90)
    print("CONTOURLET-STYLE IMAGE COMPRESSION EXPERIMENT")
    print("=" * 90)
    print(f"Image                : {image_path}")
    print(f"Levels               : {levels}")
    print(f"Directions per level : {directions_per_level}")
    print(f"Image shape          : {image.shape}")
    print("-" * 90)

    results = []

    for keep_ratio in keep_ratios:
        result = compress_contourlet_topk(
            image=image,
            levels=levels,
            directions_per_level=directions_per_level,
            keep_ratio=keep_ratio
        )

        reconstructed = result["reconstructed"]

        tag = f"keep_{int(keep_ratio * 100):02d}"
        recon_path = output_dir / f"reconstructed_{tag}.png"
        fig_path = output_dir / f"comparison_{tag}.png"

        save_image(reconstructed, str(recon_path))
        save_comparison_figure(
            original=image,
            reconstructed=reconstructed,
            save_path=str(fig_path),
            title=f"Contourlet-style, levels={levels}, dirs={directions_per_level}, keep={keep_ratio:.0%}"
        )

        results.append({
            "keep_ratio_target": keep_ratio,
            "keep_ratio_actual": result["actual_keep_ratio"],
            "approx_CR": result["approx_compression_ratio"],
            "MSE": result["mse"],
            "PSNR": result["psnr"],
            "SSIM": result["ssim"],
        })

    print(f"{'Target keep':>12} | {'Actual keep':>12} | {'Approx CR':>10} | {'MSE':>12} | {'PSNR':>10} | {'SSIM':>10}")
    print("-" * 90)
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

    print("-" * 90)
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
output_dir = os.path.join(BASE_DIR, "output_contourlet")

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

show_8_images(image_paths, window_title="Contourlet Output")

if __name__ == "__main__":
    run_experiment()