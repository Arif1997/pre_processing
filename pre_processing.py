import cv2
import numpy as np
import base64
from skimage.metrics import structural_similarity as ssim
import os

def apply_denoising_filter(image):
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised

def apply_mean_filter(image, kernel_size=(5, 5)):
    mean_filtered = cv2.blur(image, kernel_size)
    return mean_filtered

def apply_median_filter(image, kernel_size=5):
    median_filtered = cv2.medianBlur(image, kernel_size)
    return median_filtered

def apply_edge_detection_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def apply_laplacian_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

def apply_morphological_filter(image, operation='open', kernel_size=(5, 5)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    if operation == 'open':
        morph_filtered = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        morph_filtered = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    elif operation == 'erode':
        morph_filtered = cv2.erode(image, kernel)
    elif operation == 'dilate':
        morph_filtered = cv2.dilate(image, kernel)
    else:
        morph_filtered = image
    return morph_filtered

def apply_sobel_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.hypot(sobelx, sobely)
    return cv2.convertScaleAbs(sobel)

def apply_brightness_filter(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def apply_contrast_filter(image, alpha=1.5):
    contrast_filtered = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return contrast_filtered

def apply_color_filter(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def apply_gaussian_blur_filter(image, kernel_size=(5, 5)):
    gaussian_blurred = cv2.GaussianBlur(image, kernel_size, 0)
    return gaussian_blurred

def apply_inverted_filter(image):
    inverted = cv2.bitwise_not(image)
    return inverted

def apply_sharpening_filter(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def apply_resized_filter(image, size=(100, 100)):
    resized = cv2.resize(image, size)
    return resized

def apply_scaled_filter(image, fx=1.5, fy=1.5):
    scaled = cv2.resize(image, None, fx=fx, fy=fy)
    return scaled

def apply_detail_filter(image):
    detail = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
    return detail

def apply_edge_enhance_filter(image):
    edge_enhanced = cv2.edgePreservingFilter(image, flags=1, sigma_s=60, sigma_r=0.4)
    return edge_enhanced

def apply_equalized_filter(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    ycrcb = cv2.merge(channels)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def calculate_ssim(original, filtered):
    # Ensure images are the same size and type
    if original.shape != filtered.shape:
        filtered = cv2.resize(filtered, (original.shape[1], original.shape[0]))
    if original.dtype != filtered.dtype:
        filtered = filtered.astype(original.dtype)

    # Convert to grayscale if the images are color
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        filtered_gray = filtered

    ssim_value, _ = ssim(original_gray, filtered_gray, full=True)

    return ssim_value * 100  # Convert to percentage

def calculate_psnr(original, filtered):
    # Ensure images are the same size and type
    if original.shape != filtered.shape:
        filtered = cv2.resize(filtered, (original.shape[1], original.shape[0]))
    if original.dtype != filtered.dtype:
        filtered = filtered.astype(original.dtype)

    mse = np.mean((original - filtered) ** 2)
    if mse == 0:
        return 100  # Avoid division by zero, maximum PSNR
    PIXEL_MAX = 255.0
    psnr_value = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr_value  # PSNR is already in dB


def calculate_metrics_for_folder(folder_path):
    images = os.listdir(folder_path)
    ssim_results = []
    psnr_results = []
    metrics_averages = {
        'Denoised': {'SSIM': [], 'PSNR': []},
        'Mean Filtered': {'SSIM': [], 'PSNR': []},
        'Median Filtered': {'SSIM': [], 'PSNR': []},
        'Edge Detection': {'SSIM': [], 'PSNR': []},
        'Laplacian': {'SSIM': [], 'PSNR': []},
        'Morphological Filter': {'SSIM': [], 'PSNR': []},
        'Sobel Filter': {'SSIM': [], 'PSNR': []},
        'Brightness': {'SSIM': [], 'PSNR': []},
        'Contrast': {'SSIM': [], 'PSNR': []},
        'Color': {'SSIM': [], 'PSNR': []},
        'Gaussian Blur': {'SSIM': [], 'PSNR': []},
        'Inverted': {'SSIM': [], 'PSNR': []},
        'Sharpening': {'SSIM': [], 'PSNR': []},
        'Resized': {'SSIM': [], 'PSNR': []},
        'Scaled': {'SSIM': [], 'PSNR': []},
        'Detail': {'SSIM': [], 'PSNR': []},
        'Edge Enhance': {'SSIM': [], 'PSNR': []},
        'Equalized': {'SSIM': [], 'PSNR': []}
    }

    output_folder_base = 'D:/Projects/pre_processing/result'

    for image_name in images:
        image_path = os.path.join(folder_path, image_name)
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Ensure the image has a valid extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_name_with_ext = image_name if any(image_name.lower().endswith(ext) for ext in valid_extensions) else image_name + '.jpg'

        if original_image is not None:
            print(f"Processing image: {image_name}")

            # Apply filters and calculate SSIM and PSNR
            filters = {
                'Denoised': apply_denoising_filter,
                'Mean Filtered': apply_mean_filter,
                'Median Filtered': apply_median_filter,
                'Edge Detection': apply_edge_detection_filter,
                'Laplacian': apply_laplacian_filter,
                'Morphological Filter': lambda img: apply_morphological_filter(img, operation='open'),
                'Sobel Filter': apply_sobel_filter,
                'Brightness': apply_brightness_filter,
                'Contrast': apply_contrast_filter,
                'Color': apply_color_filter,
                'Gaussian Blur': apply_gaussian_blur_filter,
                'Inverted': apply_inverted_filter,
                'Sharpening': apply_sharpening_filter,
                'Resized': apply_resized_filter,
                'Scaled': apply_scaled_filter,
                'Detail': apply_detail_filter,
                'Edge Enhance': apply_edge_enhance_filter,
                'Equalized': apply_equalized_filter
            }

            for filter_name, filter_func in filters.items():
                filtered_image = filter_func(original_image.copy())

                # Save the filtered image
                filter_output_folder = os.path.join(output_folder_base, filter_name)
                os.makedirs(filter_output_folder, exist_ok=True)
                output_image_path = os.path.join(filter_output_folder, image_name_with_ext)
                cv2.imwrite(output_image_path, filtered_image)

                if filter_name in ['Edge Detection', 'Laplacian', 'Sobel Filter']:
                    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                    ssim_value = calculate_ssim(original_gray, filtered_image)
                    psnr_value = calculate_psnr(original_gray, filtered_image)
                else:
                    ssim_value = calculate_ssim(original_image, filtered_image)
                    psnr_value = calculate_psnr(original_image, filtered_image)

                ssim_results.append({
                    'Image': image_name,
                    'Filter': filter_name,
                    'SSIM': ssim_value,
                    'PSNR': psnr_value
                })

                psnr_results.append({
                    'Image': image_name,
                    'Filter': filter_name,
                    'SSIM': ssim_value,
                    'PSNR': psnr_value
                })

                metrics_averages[filter_name]['SSIM'].append(ssim_value)
                metrics_averages[filter_name]['PSNR'].append(psnr_value)


    # Calculate average SSIM and PSNR values for each filter
    avg_metrics = {
        filter_name: {
            'SSIM': np.mean(values['SSIM']),
            'PSNR': np.mean(values['PSNR'])
        } for filter_name, values in metrics_averages.items()
    }

    # Determine the filter with the highest average SSIM value
    best_filter_ssim = max(avg_metrics, key=lambda k: avg_metrics[k]['SSIM'])
    best_ssim_accuracy = avg_metrics[best_filter_ssim]['SSIM']

    # Determine the filter with the highest average PSNR value
    best_filter_psnr = max(avg_metrics, key=lambda k: avg_metrics[k]['PSNR'])
    best_psnr_accuracy = avg_metrics[best_filter_psnr]['PSNR']
    return best_filter_ssim, best_ssim_accuracy, best_filter_psnr, best_psnr_accuracy





def image_to_base64(image):
        _, buffer = cv2.imencode('.jpg', image)  # Convert to JPEG format
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64

def calculate_metrics_for_image(image):
        ssim_results = []
        psnr_results = []
        metrics_averages = {
            'Denoised': {'SSIM': [], 'PSNR': []},
            'Mean Filtered': {'SSIM': [], 'PSNR': []},
            'Median Filtered': {'SSIM': [], 'PSNR': []},
            'Edge Detection': {'SSIM': [], 'PSNR': []},
            'Laplacian': {'SSIM': [], 'PSNR': []},
            'Morphological Filter': {'SSIM': [], 'PSNR': []},
            'Sobel Filter': {'SSIM': [], 'PSNR': []},
            'Brightness': {'SSIM': [], 'PSNR': []},
            'Contrast': {'SSIM': [], 'PSNR': []},
            'Color': {'SSIM': [], 'PSNR': []},
            'Gaussian Blur': {'SSIM': [], 'PSNR': []},
            'Inverted': {'SSIM': [], 'PSNR': []},
            'Sharpening': {'SSIM': [], 'PSNR': []},
            'Resized': {'SSIM': [], 'PSNR': []},
            'Scaled': {'SSIM': [], 'PSNR': []},
            'Detail': {'SSIM': [], 'PSNR': []},
            'Edge Enhance': {'SSIM': [], 'PSNR': []},
            'Equalized': {'SSIM': [], 'PSNR': []}
        }

        # Ensure the image is valid
        if image is not None:
            print(f"Processing received image.")

            # Apply filters and calculate SSIM and PSNR
            filters = {
                'Denoised': apply_denoising_filter,
                'Mean Filtered': apply_mean_filter,
                'Median Filtered': apply_median_filter,
                'Edge Detection': apply_edge_detection_filter,
                'Laplacian': apply_laplacian_filter,
                'Morphological Filter': lambda img: apply_morphological_filter(img, operation='open'),
                'Sobel Filter': apply_sobel_filter,
                'Brightness': apply_brightness_filter,
                'Contrast': apply_contrast_filter,
                'Color': apply_color_filter,
                'Gaussian Blur': apply_gaussian_blur_filter,
                'Inverted': apply_inverted_filter,
                'Sharpening': apply_sharpening_filter,
                'Resized': apply_resized_filter,
                'Scaled': apply_scaled_filter,
                'Detail': apply_detail_filter,
                'Edge Enhance': apply_edge_enhance_filter,
                'Equalized': apply_equalized_filter
            }

            best_ssim_image = None
            best_psnr_image = None
            best_ssim_value = -1  # Initialize to a very low value
            best_psnr_value = -1  # Initialize to a very low value

            for filter_name, filter_func in filters.items():
                filtered_image = filter_func(image.copy())

                if filter_name in ['Edge Detection', 'Laplacian', 'Sobel Filter']:
                    original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    ssim_value = calculate_ssim(original_gray, filtered_image)
                    psnr_value = calculate_psnr(original_gray, filtered_image)
                else:
                    ssim_value = calculate_ssim(image, filtered_image)
                    psnr_value = calculate_psnr(image, filtered_image)

                ssim_results.append({
                    'Filter': filter_name,
                    'SSIM': ssim_value,
                    'PSNR': psnr_value
                })

                psnr_results.append({
                    'Filter': filter_name,
                    'SSIM': ssim_value,
                    'PSNR': psnr_value
                })

                metrics_averages[filter_name]['SSIM'].append(ssim_value)
                metrics_averages[filter_name]['PSNR'].append(psnr_value)

                # Update the best SSIM image
                if ssim_value > best_ssim_value:
                    best_ssim_value = ssim_value
                    best_ssim_image = filtered_image

                # Update the best PSNR image
                if psnr_value > best_psnr_value:
                    best_psnr_value = psnr_value
                    best_psnr_image = filtered_image

        # Calculate average SSIM and PSNR values for each filter
        avg_metrics = {
            filter_name: {
                'SSIM': np.mean(values['SSIM']),
                'PSNR': np.mean(values['PSNR'])
            } for filter_name, values in metrics_averages.items()
        }

        # Determine the filter with the highest average SSIM value
        best_filter_ssim = max(avg_metrics, key=lambda k: avg_metrics[k]['SSIM'])
        best_ssim_accuracy = avg_metrics[best_filter_ssim]['SSIM']

        # Determine the filter with the highest average PSNR value
        best_filter_psnr = max(avg_metrics, key=lambda k: avg_metrics[k]['PSNR'])
        best_psnr_accuracy = avg_metrics[best_filter_psnr]['PSNR']
        
        best_ssim_image_base64 = image_to_base64(best_ssim_image)
        best_psnr_image_base64 = image_to_base64(best_psnr_image)

        
        return (
            best_filter_ssim,
            best_ssim_accuracy,
            best_filter_psnr,
            best_psnr_accuracy,
            best_psnr_image_base64,
            best_ssim_image_base64, 

        )


def apply_filter_and_calculate_metrics(image, filter_name):
    # Dictionary of filter functions
    filters = {
        'Denoised': apply_denoising_filter,
        'Mean Filtered': apply_mean_filter,
        'Median Filtered': apply_median_filter,
        'Edge Detection': apply_edge_detection_filter,
        'Laplacian': apply_laplacian_filter,
        'Morphological Filter': lambda img: apply_morphological_filter(img, operation='open'),
        'Sobel Filter': apply_sobel_filter,
        'Brightness': apply_brightness_filter,
        'Contrast': apply_contrast_filter,
        'Color': apply_color_filter,
        'Gaussian Blur': apply_gaussian_blur_filter,
        'Inverted': apply_inverted_filter,
        'Sharpening': apply_sharpening_filter,
        'Resized': apply_resized_filter,
        'Scaled': apply_scaled_filter,
        'Detail': apply_detail_filter,
        'Edge Enhance': apply_edge_enhance_filter,
        'Equalized': apply_equalized_filter
    }

    # Check if the filter name is valid
    if filter_name not in filters:
        raise ValueError(f"Filter '{filter_name}' is not a valid filter name.")

    # Apply the selected filter
    filter_func = filters[filter_name]
    filtered_image = filter_func(image.copy())

    # Calculate SSIM and PSNR
    if filter_name in ['Edge Detection', 'Laplacian', 'Sobel Filter']:
        original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ssim_value = calculate_ssim(original_gray, filtered_image)
        psnr_value = calculate_psnr(original_gray, filtered_image)
    else:
        ssim_value = calculate_ssim(image, filtered_image)
        psnr_value = calculate_psnr(image, filtered_image)
    
    best_ssim_image_base64 = image_to_base64(filtered_image)
    best_psnr_image_base64 = image_to_base64(filtered_image)


    return ssim_value, psnr_value, best_psnr_image_base64, best_ssim_image_base64
