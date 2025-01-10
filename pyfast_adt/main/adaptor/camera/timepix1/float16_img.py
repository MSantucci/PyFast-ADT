import cv2
import imageio
import tifffile as tiff
import numpy as np
import glob

# Path to the folder containing the TIFF files
folder_path = "test_frames/"

# Get a list of TIFF file paths
tif_paths = glob.glob(folder_path + '*.tiff')

# Process each TIFF file
for tif_path in tif_paths:
    # Load the TIFF file using tifffile
    tiff_data = tiff.imread(tif_path)

    # Convert the data to floating-point 16-bit format
    float_data = tiff_data.astype(np.float16)

    # Save the floating-point TIFF file using OpenCV
    float_path = tif_path.replace('.tif', '_float.tif')
    imageio.imwrite(float_path, tiff_data.astype(np.float16), format="tiff")

    print(f"Converted {tif_path} to floating-point format and saved as {float_path}")
