import numpy as np
import sys
import getopt

from dicom_loader import save_dicom_directory, load_scans
from utils import find_lungs, intensity_seg, create_mask_from_polygon, create_vessel_mask, compute_area, plot_image, \
    set_manual_window

opts, args = getopt.getopt(sys.argv[1:], "hdp:s:l:w:vi:",
                           ["help", "dicom", "path=", "subject=", "level=", "window=", "visualize", "slice="])

# Flag to know if script should load the dicom directory locally
LOAD_DICOM_DIR = False
# We assume the scans are two folders back
dicom_dir_path = '../SCAN_2/DICOMDIR'

# Patient holding the scans we want to obtain
subject = ""

# Hounds field unit parameters
hu_level = -100
hu_window = 300

SHOULD_VISUALIZE = False
scan_slice = 100

for opt, arg in opts:
    if opt in ("-h", "--help"):
        print("For help use:")
        print("{0} -h".format(sys.argv[0]))
        print("To load a dicom directory use:")
        print("{0} -d -p <path>".format(sys.argv[0]))
        print("To load scans and compute the density use:")
        print("{0} -s <patient name> -l <level> -w <window>".format(sys.argv[0]))
        print("To visualize a specific slice use:")
        print("{0} -v -sl <slice number>".format(sys.argv[0]))
        sys.exit(2)
    elif opt in ("-d", "--dicom"):
        LOAD_DICOM_DIR = True
    elif opt in ("-p", "--path"):
        dicom_dir_path = arg
    elif opt in ("-s", "--subject"):
        subject = arg
    elif opt in ("-l", "--level"):
        level = arg
    elif opt in ("-w", "--window"):
        window = arg
    elif opt in ("-v", "--visualize"):
        SHOULD_VISUALIZE = True
    elif opt in ("-i", "--slice"):
        scan_slice = int(arg)

if __name__ == '__main__':

    # returns a dictionary where the key is the patient id and the value is the scans
    if LOAD_DICOM_DIR:
        save_dicom_directory(dicom_dir_path)
    else:
        images, pixel_dimensions = load_scans(subject)

        densities = []
        total_density = 0

        total_images = len(images)

        # Image masked that should be displayed
        mask_to_display = []

        for i in range(len(images)):
            contours = intensity_seg(images[i], hu_level, hu_window)
            contours = find_lungs(contours)

            # if we cant find any contours on the image
            if contours is None:
                continue

            lung_mask = create_mask_from_polygon(images[i], contours)
            vessel_mask = np.array(create_vessel_mask(lung_mask, images[i], hu_level, hu_window), dtype=np.uint8)

            lung_area = compute_area(lung_mask, pixel_dimensions[i])
            vessel_area = compute_area(vessel_mask, pixel_dimensions[i])

            current_density = 100 - (vessel_area / lung_area * 100)
            densities.append(current_density)
            total_density += current_density

            print("Calculating Densities. %.2f%% Completed... Slice Number: %d. Current Avg Density %.2f%%. Slice Density %.2f%%" % ((i * 100 / total_images), i, (total_density / (i + 1)), current_density))

        if SHOULD_VISUALIZE:
            plot_image(set_manual_window(images[scan_slice], hu_level, hu_window), mask_to_display)
