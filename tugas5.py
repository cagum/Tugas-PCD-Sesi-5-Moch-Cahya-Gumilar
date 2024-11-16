import imageio
import numpy as np
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt

def apply_filter(image, filter_type):

    if filter_type == 'low-pass':
        return gaussian_filter(image, sigma=2)
    
    elif filter_type == 'high-pass':
        if len(image.shape) == 3:
            image_gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            image_gray = image
        sobel_x = sobel(image_gray, axis=0)
        sobel_y = sobel(image_gray, axis=1)
        edges = np.hypot(sobel_x, sobel_y)
        edges = (edges / np.max(edges) * 255).astype(np.uint8)
        return edges
    
    elif filter_type == 'high-boost':
        if len(image.shape) == 3:
            image_gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])  
        else:
            image_gray = image
        sobel_x = sobel(image_gray, axis=0)
        sobel_y = sobel(image_gray, axis=1)
        edges = np.hypot(sobel_x, sobel_y)
        edges = (edges / np.max(edges) * 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            edges = np.stack([edges] * 3, axis=-1)

        return np.clip(image + edges, 0, 255).astype(np.uint8)

def plot_results(original, low_pass, high_pass, high_boost):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(low_pass, cmap='gray')
    plt.title("Low-pass Filter (Gaussian Blur)")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(high_pass, cmap='gray')
    plt.title("High-pass Filter (Sobel Edge Detection)")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(high_boost)
    plt.title("High-boost Filter")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    
    image_path = "D:\Perkuliahan\S5\Pengolahan Citra Digital\s4\praktikum tugas 4\capmar.jpg"  # Ganti dengan path gambar Anda
    original_image = imageio.imread(image_path)

    if len(original_image.shape) == 3:
        grayscale_image = np.dot(original_image[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        grayscale_image = original_image

    low_pass_grayscale = apply_filter(grayscale_image, 'low-pass')
    high_pass_grayscale = apply_filter(grayscale_image, 'high-pass')
    high_boost_grayscale = apply_filter(grayscale_image, 'high-boost')

    low_pass_color = apply_filter(original_image, 'low-pass')
    high_pass_color = apply_filter(original_image, 'high-pass')
    high_boost_color = apply_filter(original_image, 'high-boost')

    print("Menampilkan hasil filter pada citra grayscale dan berwarna.")
    plot_results(grayscale_image, low_pass_grayscale, high_pass_grayscale, high_boost_grayscale)
    plot_results(original_image, low_pass_color, high_pass_color, high_boost_color)

if __name__ == "__main__":
    main()
