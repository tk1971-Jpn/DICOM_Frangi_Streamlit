from pathlib import Path
import numpy as np
import pydicom


def load_dicom_series(folder_path: str):
    """
    DICOMフォルダから1シリーズ分の画像を読み込み、
    3D volume (Z, Y, X) を返す。
    """
    folder = Path(folder_path)
    files = sorted([f for f in folder.iterdir() if f.is_file()])

    dicoms = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            if hasattr(ds, "PixelData"):
                dicoms.append(ds)
        except Exception:
            continue

    if len(dicoms) == 0:
        raise ValueError("PixelDataを持つDICOMファイルが見つかりませんでした。")

    def sort_key(ds):
        if hasattr(ds, "InstanceNumber"):
            return int(ds.InstanceNumber)
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2])
        return 0

    dicoms = sorted(dicoms, key=sort_key)

    slices = []
    for ds in dicoms:
        arr = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        slices.append(arr)

    volume = np.stack(slices, axis=0)  # (Z, Y, X)

    metadata = {
        "num_slices": volume.shape[0],
        "height": volume.shape[1],
        "width": volume.shape[2],
        "series_description": str(getattr(dicoms[0], "SeriesDescription", "")),
        "pixel_spacing": getattr(dicoms[0], "PixelSpacing", None),
        "slice_thickness": getattr(dicoms[0], "SliceThickness", None),
    }

    return volume, metadata