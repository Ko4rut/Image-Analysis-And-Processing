import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Utils
# =========================================================
def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa ảnh về [0, 1] để hiển thị rõ bằng matplotlib.
    """
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def read_grayscale_image(image_path: str) -> np.ndarray:
    """
    Đọc ảnh grayscale float32.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    return img.astype(np.float32)


# =========================================================
# Laplacian Pyramid
# =========================================================
def build_laplacian_pyramid(image: np.ndarray, levels: int = 3):
    """
    Trả về:
        gaussian_pyramid: [G0, G1, ..., G_levels]
        laplacian_pyramid: [L0, L1, ..., L_{levels-1}, G_levels]
    Trong đó:
        Li = Gi - upsample(G{i+1})
        phần tử cuối cùng là lowpass residual G_levels
    """
    gaussian_pyramid = [image]
    laplacian_pyramid = []

    current = image.copy()
    for _ in range(levels):
        down = cv2.pyrDown(current)
        gaussian_pyramid.append(down)
        current = down

    for i in range(levels):
        g_current = gaussian_pyramid[i]
        g_next = gaussian_pyramid[i + 1]

        up = cv2.pyrUp(g_next)
        if up.shape != g_current.shape:
            up = cv2.resize(up, (g_current.shape[1], g_current.shape[0]))

        lap = g_current - up
        laplacian_pyramid.append(lap)

    # thêm lowpass cuối
    laplacian_pyramid.append(gaussian_pyramid[-1])

    return gaussian_pyramid, laplacian_pyramid


# =========================================================
# Directional decomposition (practical approximation)
# Dùng Gabor filters để tạo nhiều hướng rõ ràng.
# =========================================================
def create_gabor_kernels(
    num_directions: int = 8,
    ksize: int = 21,
    sigma: float = 5.0,
    lambd: float = 10.0,
    gamma: float = 0.5,
    psi: float = 0
):
    """
    Tạo bank Gabor kernel theo nhiều hướng.
    """
    kernels = []
    angles = []

    for i in range(num_directions):
        theta = np.pi * i / num_directions
        kernel = cv2.getGaborKernel(
            (ksize, ksize),
            sigma=sigma,
            theta=theta,
            lambd=lambd,
            gamma=gamma,
            psi=psi,
            ktype=cv2.CV_32F
        )

        # chuẩn hóa để tránh band nào quá mạnh, band nào quá yếu
        norm = np.sum(np.abs(kernel))
        if norm > 1e-8:
            kernel = kernel / norm

        kernels.append(kernel)
        angles.append(theta)

    return kernels, angles


def directional_decompose(
    band: np.ndarray,
    num_directions: int = 8,
    ksize: int = 21,
    sigma: float = 5.0,
    lambd: float = 10.0,
    gamma: float = 0.5
):
    """
    Phân rã 1 bandpass thành nhiều hướng bằng Gabor filter bank.
    Trả về danh sách directional subbands.
    """
    kernels, angles = create_gabor_kernels(
        num_directions=num_directions,
        ksize=ksize,
        sigma=sigma,
        lambd=lambd,
        gamma=gamma
    )

    subbands = []
    for kernel in kernels:
        filtered = cv2.filter2D(band, cv2.CV_32F, kernel)
        subbands.append(filtered)

    return subbands, angles


# =========================================================
# Visualization
# =========================================================
def show_pyramid(gaussian_pyramid, laplacian_pyramid):
    levels = len(gaussian_pyramid) - 1

    plt.figure(figsize=(4 * (levels + 1), 4))
    for i, g in enumerate(gaussian_pyramid):
        plt.subplot(1, levels + 1, i + 1)
        plt.imshow(normalize_for_display(g), cmap="gray")
        plt.title(f"G{i}\n{g.shape[1]}x{g.shape[0]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(4 * len(laplacian_pyramid), 4))
    for i, l in enumerate(laplacian_pyramid[:-1]):
        plt.subplot(1, len(laplacian_pyramid), i + 1)
        plt.imshow(normalize_for_display(l), cmap="gray")
        plt.title(f"L{i}\n{l.shape[1]}x{l.shape[0]}")
        plt.axis("off")

    plt.subplot(1, len(laplacian_pyramid), len(laplacian_pyramid))
    plt.imshow(normalize_for_display(laplacian_pyramid[-1]), cmap="gray")
    plt.title(f"Lowpass\n{laplacian_pyramid[-1].shape[1]}x{laplacian_pyramid[-1].shape[0]}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_directional_subbands(all_directional, num_directions=8):
    """
    all_directional: list, mỗi phần tử là list subbands của 1 mức Laplacian
    """
    for level_idx, subbands in enumerate(all_directional):
        rows = 2
        cols = int(np.ceil(num_directions / rows))
        plt.figure(figsize=(4 * cols, 4 * rows))

        for j, band in enumerate(subbands):
            plt.subplot(rows, cols, j + 1)
            plt.imshow(normalize_for_display(band), cmap="gray")
            plt.title(f"L{level_idx} - Dir{j}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()


def save_directional_results(output_dir, all_directional):
    os.makedirs(output_dir, exist_ok=True)

    for level_idx, subbands in enumerate(all_directional):
        level_dir = os.path.join(output_dir, f"level_{level_idx}")
        os.makedirs(level_dir, exist_ok=True)

        for dir_idx, band in enumerate(subbands):
            vis = (normalize_for_display(band) * 255).astype(np.uint8)
            out_path = os.path.join(level_dir, f"dir_{dir_idx}.png")
            cv2.imwrite(out_path, vis)


# =========================================================
# Main
# =========================================================
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(BASE_DIR, "..", "..", "Images", "lena_std.tif")

    # Nếu path trên không tồn tại, đổi tay ở đây
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Không tìm thấy ảnh ở: {image_path}\n"
            f"Hãy sửa lại biến image_path cho đúng."
        )

    img = read_grayscale_image(image_path)

    # 1) LP
    gaussian_pyramid, laplacian_pyramid = build_laplacian_pyramid(img, levels=3)

    # Có thể bật nếu muốn xem LP
    show_pyramid(gaussian_pyramid, laplacian_pyramid)

    # 2) Directional decomposition trên từng bandpass của LP
    laplacian_bands = laplacian_pyramid[:-1]   # bỏ lowpass cuối
    all_directional = []

    for i, lap in enumerate(laplacian_bands):
        # Ở scale thô hơn nên tăng wavelength một chút để bắt cấu trúc phù hợp hơn
        if i == 0:
            lambd = 8.0
            sigma = 4.0
            ksize = 17
        elif i == 1:
            lambd = 10.0
            sigma = 5.0
            ksize = 21
        else:
            lambd = 12.0
            sigma = 6.0
            ksize = 25

        subbands, _ = directional_decompose(
            lap,
            num_directions=8,
            ksize=ksize,
            sigma=sigma,
            lambd=lambd,
            gamma=0.5
        )

        all_directional.append(subbands)

    # 3) Hiển thị
    show_directional_subbands(all_directional, num_directions=8)

    # 4) Lưu kết quả nếu cần
    output_dir = os.path.join(BASE_DIR, "output_contourlet_style")
    save_directional_results(output_dir, all_directional)
    print(f"Đã lưu subbands vào: {output_dir}")

if __name__ == "__main__":
    main()