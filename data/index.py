FILTERS_INFO = {
    "Denoised": {
        "name": "Denoised Filter",
        "description": "Reduces noise from the image, typically useful for removing grain or random variation in pixel intensity caused by low-light photography or sensor noise. This filter smooths out unnecessary variations while preserving essential details of the image."
    },
    "Mean Filtered": {
        "name": "Mean Filter",
        "description": "Applies a mean filter, which is a basic smoothing technique that replaces each pixel value with the average of its surrounding pixels. It's useful for reducing noise, but can also blur edges in the process."
    },
    "Median Filtered": {
        "name": "Median Filter",
        "description": "Applies a median filter, particularly effective in removing 'salt and pepper' noise. Each pixel is replaced with the median value from its neighbors, preserving edges while eliminating extreme outliers."
    },
    "Edge Detection": {
        "name": "Edge Detection Filter",
        "description": "Detects edges in the image by identifying areas of high contrast or rapid intensity changes. This filter is essential for highlighting boundaries, making it useful for image segmentation and object detection."
    },
    "Laplacian": {
        "name": "Laplacian Filter",
        "description": "Applies a Laplacian filter that is used to highlight regions of rapid intensity change, effectively detecting edges. It's a second-order derivative operator, which calculates the rate of change of pixel intensity, making it great for edge enhancement."
    },
    "Morphological Filter": {
        "name": "Morphological Filter (Open Operation)",
        "description": "Performs a morphological 'open' operation, which removes small noise by first eroding the image and then dilating it. This operation is useful for cleaning up images and removing small artifacts or details without significantly altering larger structures."
    },
    "Sobel Filter": {
        "name": "Sobel Filter",
        "description": "Applies the Sobel filter, an edge detection operator that calculates the gradient of image intensity at each pixel, allowing for the detection of edges in both horizontal and vertical directions. It’s particularly useful for detecting edges along certain orientations."
    },
    "Brightness": {
        "name": "Brightness Adjustment",
        "description": "Increases or decreases the brightness of the image by uniformly adjusting pixel intensity values. This can be useful in correcting underexposed or overexposed images, making details more visible."
    },
    "Contrast": {
        "name": "Contrast Adjustment",
        "description": "Modifies the contrast of the image by increasing or decreasing the difference between light and dark areas. Enhanced contrast can make objects in the image more distinguishable and vivid, improving clarity."
    },
    "Color": {
        "name": "Color Adjustment",
        "description": "Adjusts the color properties of the image by modifying hue, saturation, and intensity. This filter can enhance specific colors or overall tonal quality, making the image appear more vibrant or subdued."
    },
    "Gaussian Blur": {
        "name": "Gaussian Blur",
        "description": "Applies a Gaussian blur, a type of image-blurring filter that reduces image noise and detail by averaging pixel values in a manner weighted by their distance from the center. It provides a soft blur and is commonly used in preprocessing to reduce noise and detail."
    },
    "Inverted": {
        "name": "Inverted Filter",
        "description": "Inverts the colors of the image, replacing each pixel value with its complementary color. This filter can create a dramatic effect and is often used for artistic or diagnostic purposes to highlight certain features."
    },
    "Sharpening": {
        "name": "Sharpening Filter",
        "description": "Enhances the sharpness of the image by emphasizing the edges, making details more distinct. This filter is used to improve the clarity of objects within an image, giving it a more defined and crisp appearance."
    },
    "Resized": {
        "name": "Resized Filter",
        "description": "Resizes the image by scaling its dimensions, either enlarging or reducing the image size. This operation is essential for fitting images into different formats or for preparing images for specific tasks such as classification or recognition."
    },
    "Scaled": {
        "name": "Scaled Filter",
        "description": "Alters the scaling of the image, either by upscaling or downscaling the resolution. This operation is useful for adjusting image dimensions without introducing too much distortion or pixelation."
    },
    "Detail": {
        "name": "Detail Enhancement",
        "description": "Enhances the finer details of the image, bringing out textures and subtle features that might otherwise be overlooked. This filter is especially useful in scenarios where detecting minute details is crucial, such as in forensic imaging or medical diagnostics."
    },
    "Edge Enhance": {
        "name": "Edge Enhance Filter",
        "description": "Emphasizes the edges in an image, making transitions between different regions more distinct. This filter improves the visibility of object boundaries, making it easier to detect shapes and contours."
    },
    "Equalized": {
        "name": "Histogram Equalization",
        "description": "Applies histogram equalization, a technique that enhances the contrast of the image by spreading out the most frequent intensity values. This results in a more balanced image with improved contrast across different regions."
    }
}
