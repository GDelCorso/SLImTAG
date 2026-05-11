# -*- coding: utf-8 -*-

"""
Unified medical volume loader.

Supported:
    - DICOM (.dcm or directory)
    - NRRD (.nrrd / .nhdr)
    - NIfTI (.nii / .nii.gz)

Returns:
    metadata (dict)
    spacing (dx, dy, dz)
    volume (3D numpy array)

Author: Giulio Del Corso
"""


#%% Basic Libraries
import os
import numpy as np

#%% Medical libraries
import pydicom        # DICOM
import nrrd           # NRRD
import nibabel as nib # NIFTI



#%% Main auto-loader function (DICOM/NRRD/NIFTI))
def load_medical_volume(path):

    # If a DICOM directory is provided
    if os.path.isdir(path):
        return load_DICOM(path)

    
    lower = path.lower()

    # NRRD
    if lower.endswith(".nrrd") or lower.endswith(".nhdr"):
        return load_NRRD(path)

    # NIFTI
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        return load_NIFTI(path)

    # DICOM
    if lower.endswith(".dcm"):
        return load_DICOM(path)

    # Try DICOM fallback
    try:
        return load_DICOM(path)

    except Exception:
        pass

    raise ValueError(f"Unsupported file format: {path}")
    


#%% DICOM utilities
def get_spacing_DICOM(ds):
    """
    In-plane spacing (dx, dy).
    """

    if "PixelSpacing" in ds:
        dy, dx = map(float, ds.PixelSpacing)
        return dx, dy

    if "ImagerPixelSpacing" in ds:
        dy, dx = map(float, ds.ImagerPixelSpacing)
        return dx, dy

    if "PixelAspectRatio" in ds:
        y, x = map(float, ds.PixelAspectRatio)
        return x, y

    return 1.0, 1.0



def get_slice_position_DICOM(ds):

    if "ImagePositionPatient" in ds:
        return np.array(ds.ImagePositionPatient, dtype=np.float64)

    return None



def get_orientation_DICOM(ds):

    if "ImageOrientationPatient" in ds:

        o = np.array(ds.ImageOrientationPatient, dtype=np.float64)

        row = o[:3]
        col = o[3:]

        return row, col

    return None, None



def compute_slice_spacing_DICOM(sorted_slices):

    positions = []

    for s in sorted_slices:

        pos = get_slice_position_DICOM(s)

        if pos is not None:
            positions.append(pos)

    if len(positions) >= 2:

        positions = np.array(positions)

        diffs = np.linalg.norm(
            np.diff(positions, axis=0),
            axis=1
        )

        return float(np.mean(diffs))

    ds0 = sorted_slices[0]

    return float(ds0.get("SliceThickness", 1.0))



def sort_slices_DICOM(slices):

    slices_with_pos = [
        (s, get_slice_position_DICOM(s))
        for s in slices
    ]

    # fallback
    if any(p is None for _, p in slices_with_pos):

        return sorted(
            slices,
            key=lambda x: int(x.get("InstanceNumber", 0))
        )

    normal = np.array([0, 0, 1], dtype=np.float64)

    row, col = get_orientation_DICOM(slices[0])

    if row is not None and col is not None:
        normal = np.cross(row, col)

    return sorted(
        slices,
        key=lambda s: np.dot(
            get_slice_position_DICOM(s),
            normal
        )
    )


#%% load_DICOM
def load_DICOM(path):

    # -----------------------------------------------------
    # DIRECTORY (DICOM SERIES)
    # -----------------------------------------------------

    if os.path.isdir(path):

        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if not f.startswith(".")
        ]

        slices = []

        for f in files:

            try:

                ds = pydicom.dcmread(f)

                if hasattr(ds, "PixelData"):
                    slices.append(ds)

            except Exception:
                continue

        if len(slices) == 0:
            raise ValueError("No valid DICOM found")

        slices = sort_slices_DICOM(slices)

        ds0 = slices[0]

        dx, dy = get_spacing_DICOM(ds0)
        dz = compute_slice_spacing_DICOM(slices)

        spacing = (dx, dy, dz)

        metadata = {
            "format": "DICOM",
            "PatientName": str(ds0.get("PatientName", "")),
            "Modality": str(ds0.get("Modality", "")),
            "StudyDate": str(ds0.get("StudyDate", "")),
            "Rows": int(ds0.Rows),
            "Columns": int(ds0.Columns),
        }

        images = [s.pixel_array for s in slices]

        volume = np.stack(images, axis=0)

        return metadata, spacing, volume

    # -----------------------------------------------------
    # SINGLE FILE
    # -----------------------------------------------------

    ds = pydicom.dcmread(path)

    dx, dy = get_spacing_DICOM(ds)

    metadata = {
        "format": "DICOM",
        "PatientName": str(ds.get("PatientName", "")),
        "Modality": str(ds.get("Modality", "")),
        "StudyDate": str(ds.get("StudyDate", "")),
        "Rows": int(ds.Rows),
        "Columns": int(ds.Columns),
    }

    # Multi-frame
    if hasattr(ds, "NumberOfFrames") and ds.NumberOfFrames > 1:

        volume = ds.pixel_array

        dz = float(ds.get("SliceThickness", 1.0))

        spacing = (dx, dy, dz)

        return metadata, spacing, volume

    # Single slice
    volume = ds.pixel_array[np.newaxis, :, :]

    spacing = (dx, dy, 1.0)

    return metadata, spacing, volume



#%% NRRD Utilities
def get_spacing_NRRD(header):

    if "space directions" in header:

        dirs = header["space directions"]

        spacing = []

        for d in dirs:

            if d is None:
                spacing.append(1.0)

            else:
                spacing.append(float(np.linalg.norm(d)))

        return tuple(spacing[:3])

    if "spacings" in header:

        s = header["spacings"]

        return tuple(float(x) for x in s[:3])

    return (1.0, 1.0, 1.0)



#%% LOAD NRRD
def load_NRRD(path):

    volume, header = nrrd.read(path)

    # Ensure 3D
    if volume.ndim == 2:
        volume = volume[np.newaxis, :, :]

    spacing = get_spacing_NRRD(header)

    metadata = {
        "format": "NRRD",
        "type": str(header.get("type", "")),
        "dimension": int(header.get("dimension", volume.ndim)),
        "sizes": tuple(header.get("sizes", volume.shape)),
        "encoding": str(header.get("encoding", "")),
        "space": str(header.get("space", "")),
    }

    return metadata, spacing, volume



#%% LOAD NIFTI
def load_NIFTI(path):

    nii = nib.load(path)

    volume = nii.get_fdata()

    # Ensure 3D
    if volume.ndim == 2:
        volume = volume[np.newaxis, :, :]

    hdr = nii.header

    spacing = tuple(float(x) for x in hdr.get_zooms()[:3])

    metadata = {
        "format": "NIFTI",
        "datatype": str(hdr.get_data_dtype()),
        "shape": tuple(volume.shape),
    }

    return metadata, spacing, volume
