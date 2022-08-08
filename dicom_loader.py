from pathlib import Path

import numpy as np
from pydicom import dcmread

SCANS_PATH = "./scans/scan_full_"
PIXEL_DIMENSIONS_PATH = "./scans/scan_"


# Loads the scans of the provided patient
def load_scans(patient):
    scans = np.load("%s%s.npy" % (SCANS_PATH, patient)).astype(np.float64)
    pixel_dimensions = np.load("%s%s.npy" % (PIXEL_DIMENSIONS_PATH, patient)).astype(np.float64)

    return scans, pixel_dimensions


# Saves the scans from the dicom path locally
def save_dicom_directory(dicom_path):
    ds = dcmread(dicom_path)
    root_dir = Path(ds.filename).resolve().parent

    # Iterate through the PATIENT records
    for patient in ds.patient_records:
        print(f"Loading {patient.PatientID}...")

        # Find all the STUDY records for the patient
        studies = [
            ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"
        ]
        for study in studies:
            # Find all the SERIES records in the study
            all_series = [
                ii for ii in study.children if ii.DirectoryRecordType == "SERIES"
            ]

            for series in all_series:
                if series.SeriesNumber != 2:
                    continue
                # Find all the IMAGE records in the series
                images = [
                    ii for ii in series.children
                    if ii.DirectoryRecordType == "IMAGE"
                ]

                # Get the absolute file path to each instance
                #   Each IMAGE contains a relative file path to the root directory
                elems = [ii["ReferencedFileID"] for ii in images]
                # Make sure the relative file path is always a list of str
                paths = [[ee.value] if ee.VM == 1 else ee.value for ee in elems]
                paths = [Path(*p) for p in paths]
                instances = [dcmread(Path(root_dir) / p) for p in paths]

                # Image preprocessing
                images = np.stack([i.pixel_array for i in instances])
                images = images.astype(np.int16)

                images[images == -2000] = 0

                pixel_dimensions = []

                for n in range(len(instances)):

                    intercept = instances[n].RescaleIntercept
                    slope = instances[n].RescaleSlope

                    pixel_dimensions.append(instances[n].PixelSpacing)

                    if slope != 1:
                        images[n] = slope * images[n].astype(np.float64)
                        images[n] = images[n].astype(np.int16)

                    images[n] += np.int16(intercept)

                images = np.array(images, dtype=np.int16)

                np.save("%s%s.npy" % (SCANS_PATH, patient.PatientID), images)
                np.save("%s%s.npy" % (PIXEL_DIMENSIONS_PATH, patient.PatientID), pixel_dimensions)
