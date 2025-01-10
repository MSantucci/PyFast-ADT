import cv2
import numpy as np

def remove_deadpixels(img, deadpixels, d=1):
    """Remove dead pixels from the images by replacing them with the average of
    neighbouring pixels."""
    d = 1
    for (i, j) in deadpixels:
        neighbours = img[i - d:i + d + 1, j - d:j + d + 1].flatten()
        img[i, j] = np.mean(neighbours)
    return img

def apply_flatfield_correction(image, flatfield):
    # Ensure both images have the same size
    if image.shape[:2] != flatfield.shape[:2]:
        raise ValueError("Image and flatfield should have the same size.")

    # Convert the images to float32 for accurate calculations
    image = image.astype(np.float32)
    flatfield = flatfield.astype(np.float32)

    # Apply flatfield correction
    corrected_image = np.divide(image, flatfield)
    corrected_image *= np.mean(flatfield)
    # Clip the values above the maximum range (65535) and normalize
    corrected_image = np.round(corrected_image).astype(np.uint16)

    return corrected_image

def correctCross(raw, factor=1):
    out = np.empty((516, 516), dtype=raw.dtype)
    out[0:256, 0:256] = raw[0:256, 0:256]
    out[0:256, 260:516] = raw[0:256, 256:512]
    out[260:516, 0:256] = raw[256:512, 0:256]
    out[260:516, 260:516] = raw[256:512, 256:512]

    """100000 loops, best of 3: 18 us per loop."""

    out[255:258, :] = out[255] / factor
    out[:, 255:258] = out[:, 255:256] / factor

    out[258:261, :] = out[260] / factor
    out[:, 258:261] = out[:, 260:261] / factor
    return out

# Read the image
image_path = "test_data/tracking_002.tiff"
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Read the flatfield image
flatfield_path = 'test_data/flatfield.tiff'
flatfield = cv2.imread(flatfield_path, cv2.IMREAD_UNCHANGED)
deadpixels = np.argwhere(flatfield == 0)

# Apply flatfield correction
corrected_image = apply_flatfield_correction(image, flatfield)
corrected_image = remove_deadpixels(corrected_image, deadpixels)
corrected_image = correctCross(corrected_image)

cv2.imwrite("ff_corr.tiff", corrected_image)
# Display the original and corrected images
cv2.imshow('Original Image', image)
cv2.imshow('Corrected Image', corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
