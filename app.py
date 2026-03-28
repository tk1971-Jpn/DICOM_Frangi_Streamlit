import os
import glob

import numpy as np
import pandas as pd
import streamlit as st
import pydicom
import matplotlib.pyplot as plt

from skimage.filters import frangi
from skimage.morphology import remove_small_objects, binary_closing, ball
from skimage import measure


# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="DICOM ROI Crop + Frangi + Centerline", layout="wide")


# =========================================================
# Session state
# =========================================================
def init_session_state():
    defaults = {
        "volume": None,
        "meta": {},
        "cropped_volume": None,
        "crop_origin": (0, 0, 0),  # z0, y0, x0
        "vesselness": None,
        "binary_vessel": None,
        "largest_component": None,
        "centerline_points": None,
        "last_loaded_path": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_processing_results():
    st.session_state["cropped_volume"] = None
    st.session_state["crop_origin"] = (0, 0, 0)
    st.session_state["vesselness"] = None
    st.session_state["binary_vessel"] = None
    st.session_state["largest_component"] = None
    st.session_state["centerline_points"] = None


# =========================================================
# DICOM helpers
# =========================================================
def safe_get(ds, key, default=""):
    return getattr(ds, key, default) if hasattr(ds, key) else default


def read_dicom_series(folder):
    files = []
    for ext in ["*", "*.dcm", "*.DCM"]:
        files.extend(glob.glob(os.path.join(folder, ext)))

    dicoms = []
    for f in sorted(set(files)):
        try:
            ds = pydicom.dcmread(f, force=True)
            if hasattr(ds, "PixelData"):
                dicoms.append(ds)
        except Exception:
            pass

    if len(dicoms) == 0:
        raise ValueError("DICOMファイルが見つかりませんでした。")

    def sort_key(ds):
        if hasattr(ds, "ImagePositionPatient"):
            try:
                return float(ds.ImagePositionPatient[2])
            except Exception:
                pass
        if hasattr(ds, "InstanceNumber"):
            try:
                return int(ds.InstanceNumber)
            except Exception:
                pass
        return 0

    dicoms = sorted(dicoms, key=sort_key)

    arrs = []
    for ds in dicoms:
        img = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        img = img * slope + intercept
        arrs.append(img)

    volume = np.stack(arrs, axis=0)  # z, y, x

    first = dicoms[0]
    pixel_spacing = safe_get(first, "PixelSpacing", [1.0, 1.0])
    spacing_y = float(pixel_spacing[0])
    spacing_x = float(pixel_spacing[1])

    if len(dicoms) >= 2 and hasattr(dicoms[0], "ImagePositionPatient") and hasattr(dicoms[1], "ImagePositionPatient"):
        try:
            spacing_z = abs(float(dicoms[1].ImagePositionPatient[2]) - float(dicoms[0].ImagePositionPatient[2]))
        except Exception:
            spacing_z = float(safe_get(first, "SliceThickness", 1.0))
    else:
        spacing_z = float(safe_get(first, "SliceThickness", 1.0))

    meta = {
        "rows": volume.shape[1],
        "cols": volume.shape[2],
        "slices": volume.shape[0],
        "spacing_z": spacing_z,
        "spacing_y": spacing_y,
        "spacing_x": spacing_x,
        "patient_name": str(safe_get(first, "PatientName", "")),
        "study_date": str(safe_get(first, "StudyDate", "")),
        "series_description": str(safe_get(first, "SeriesDescription", "")),
        "modality": str(safe_get(first, "Modality", "")),
    }

    return volume, meta


# =========================================================
# Image helpers
# =========================================================
def apply_window(img, wl=60, ww=400):
    img = img.astype(np.float32)
    low = wl - ww / 2.0
    high = wl + ww / 2.0
    img = np.clip(img, low, high)
    img = (img - low) / (high - low + 1e-8)
    return img


def normalize01(img):
    img = img.astype(np.float32)
    mn = np.min(img)
    mx = np.max(img)
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def fig_to_array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(h, w, 3)
    return buf


def make_roi_overlay_image(base_img, x_min, x_max, y_min, y_max, figsize=(3.2, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(base_img, cmap="gray")
    rect_x = [x_min, x_max, x_max, x_min, x_min]
    rect_y = [y_min, y_min, y_max, y_max, y_min]
    ax.plot(rect_x, rect_y, linewidth=2)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    arr = fig_to_array(fig)
    plt.close(fig)
    return arr


def make_points_overlay_image(base_img, points_xy, figsize=(3.2, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(base_img, cmap="gray")
    if len(points_xy) > 0:
        xs = [p[0] for p in points_xy]
        ys = [p[1] for p in points_xy]
        ax.scatter(xs, ys, s=20)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    arr = fig_to_array(fig)
    plt.close(fig)
    return arr


# =========================================================
# Vessel processing
# =========================================================
def normalize_ct_for_frangi(volume):
    return normalize01(volume.astype(np.float32))


def run_frangi_3d(cropped_volume, sigmas=(1.0, 2.0, 3.0), gamma=15.0, threshold_percentile=90, min_object_size=100):
    vol = normalize_ct_for_frangi(cropped_volume)

    vesselness = frangi(
        vol,
        sigmas=sigmas,
        alpha=0.5,
        beta=0.5,
        gamma=gamma,
        black_ridges=False,
    )

    positive = vesselness[vesselness > 0]
    thr = np.percentile(positive, threshold_percentile) if positive.size > 0 else 0.0
    binary = vesselness > thr

    binary = remove_small_objects(binary, min_size=min_object_size)
    binary = binary_closing(binary, footprint=ball(1))

    labels = measure.label(binary, connectivity=3)

    if labels.max() > 0:
        regions = measure.regionprops(labels)
        largest_region = max(regions, key=lambda r: r.area)
        largest_component = labels == largest_region.label
    else:
        largest_component = np.zeros_like(binary, dtype=bool)

    return vesselness, binary, labels, largest_component


def extract_centerline_candidate_points(mask_3d, crop_origin=(0, 0, 0)):
    z0, y0, x0 = crop_origin
    points = []

    for z in range(mask_3d.shape[0]):
        ys, xs = np.where(mask_3d[z] > 0)
        if len(xs) == 0:
            continue

        cx = float(np.mean(xs))
        cy = float(np.mean(ys))

        points.append({
            "z_local": z,
            "y_local": cy,
            "x_local": cx,
            "z_global": z + z0,
            "y_global": cy + y0,
            "x_global": cx + x0,
        })

    return points


def crop_volume(volume, z_min, z_max, y_min, y_max, x_min, x_max):
    return volume[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]


# =========================================================
# Safe slice navigator
# =========================================================
def safe_slice_nav(key_prefix, max_index, label="Slice"):
    value_key = f"{key_prefix}_value"

    if value_key not in st.session_state:
        st.session_state[value_key] = max_index // 2

    c1, c2, c3 = st.columns([1, 4, 1])

    with c1:
        prev_clicked = st.button("◀", key=f"{key_prefix}_prev", use_container_width=True)

    with c2:
        current_val = st.number_input(
            label,
            min_value=0,
            max_value=max_index,
            value=int(st.session_state[value_key]),
            step=1,
            key=f"{key_prefix}_number_input",
        )

    with c3:
        next_clicked = st.button("▶", key=f"{key_prefix}_next", use_container_width=True)

    current_val = int(current_val)

    if prev_clicked:
        current_val = max(0, current_val - 1)

    if next_clicked:
        current_val = min(max_index, current_val + 1)

    st.session_state[value_key] = current_val

    return current_val


# =========================================================
# Main
# =========================================================
def main():
    init_session_state()

    st.title("DICOM ROI Crop + Frangi + Centerline")
    st.write("ROIを設定してcropし、そのROIに対してFrangi処理を行い、centerline候補を抽出します。")

    # -----------------------------------------------------
    # 1. DICOM load
    # -----------------------------------------------------
    st.header("1. DICOM読み込み")

    dicom_dir = st.text_input(
        "DICOMフォルダのパス",
        value="",
        placeholder="/Users/xxx/CT_series_folder"
    )

    if st.button("Load DICOM", use_container_width=True):
        try:
            if dicom_dir.strip() == "":
                st.error("DICOMフォルダのパスを入力してください。")
            elif not os.path.isdir(dicom_dir):
                st.error("指定したフォルダが存在しません。")
            else:
                volume, meta = read_dicom_series(dicom_dir)
                st.session_state["volume"] = volume
                st.session_state["meta"] = meta
                st.session_state["last_loaded_path"] = dicom_dir
                reset_processing_results()
                st.success("DICOMを読み込みました。")
        except Exception as e:
            st.error(f"読み込みエラー: {e}")

    volume = st.session_state["volume"]
    meta = st.session_state["meta"]

    if volume is None:
        st.info("上でDICOMフォルダを読み込んでください。")
        return

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------
    st.subheader("Metadata")
    meta_df = pd.DataFrame({
        "Item": [
            "PatientName",
            "StudyDate",
            "SeriesDescription",
            "Modality",
            "Volume shape (z,y,x)",
            "Spacing (z,y,x)"
        ],
        "Value": [
            meta.get("patient_name", ""),
            meta.get("study_date", ""),
            meta.get("series_description", ""),
            meta.get("modality", ""),
            str(volume.shape),
            f"({meta.get('spacing_z', 1.0):.3f}, {meta.get('spacing_y', 1.0):.3f}, {meta.get('spacing_x', 1.0):.3f}) mm"
        ]
    })
    st.dataframe(meta_df, use_container_width=True, hide_index=True)

    # -----------------------------------------------------
    # 2. Original preview
    # -----------------------------------------------------
    st.header("2. 元画像プレビュー")

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        wl = st.slider("Window Level", min_value=-300, max_value=300, value=60, step=10)
    with col_w2:
        ww = st.slider("Window Width", min_value=100, max_value=2000, value=400, step=50)

    z_idx_main = safe_slice_nav("main_slice", volume.shape[0] - 1, label="Original CT slice")
    img_main = apply_window(volume[z_idx_main], wl=wl, ww=ww)
    st.image(img_main, clamp=True, width=300)

    # -----------------------------------------------------
    # 3. ROI settings
    # -----------------------------------------------------
    st.header("3. ROI設定")

    st.write("ROI範囲を1本のスライダーで指定します。")

    z_range = st.slider(
        "z range",
        min_value=0,
        max_value=volume.shape[0] - 1,
        value=(max(0, volume.shape[0] // 3), min(volume.shape[0] - 1, volume.shape[0] * 2 // 3)),
    )
    z_min, z_max = z_range

    z_idx_roi = safe_slice_nav("roi_slice", volume.shape[0] - 1, label="ROI preview slice")
    img_roi = apply_window(volume[z_idx_roi], wl=wl, ww=ww)

    col_xy1, col_xy2 = st.columns(2)
    with col_xy1:
        x_range = st.slider(
            "x range",
            min_value=0,
            max_value=volume.shape[2] - 1,
            value=(max(0, volume.shape[2] // 4), min(volume.shape[2] - 1, volume.shape[2] * 3 // 4)),
        )
    with col_xy2:
        y_range = st.slider(
            "y range",
            min_value=0,
            max_value=volume.shape[1] - 1,
            value=(max(0, volume.shape[1] // 4), min(volume.shape[1] - 1, volume.shape[1] * 3 // 4)),
        )

    x_min, x_max = x_range
    y_min, y_max = y_range

    roi_overlay = make_roi_overlay_image(
        img_roi,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        figsize=(3.2, 3.2),
    )
    st.image(roi_overlay, caption="ROI preview", width=300)

    if st.button("Crop ROI", use_container_width=True):
        try:
            cropped_volume = crop_volume(volume, z_min, z_max, y_min, y_max, x_min, x_max)
            st.session_state["cropped_volume"] = cropped_volume
            st.session_state["crop_origin"] = (z_min, y_min, x_min)
            st.session_state["vesselness"] = None
            st.session_state["binary_vessel"] = None
            st.session_state["largest_component"] = None
            st.session_state["centerline_points"] = None
            st.success(f"ROI crop 完了: shape = {cropped_volume.shape}")
        except Exception as e:
            st.error(f"Cropエラー: {e}")

    # -----------------------------------------------------
    # 4. Crop result
    # -----------------------------------------------------
    cropped_volume = st.session_state["cropped_volume"]

    if cropped_volume is not None:
        st.header("4. Crop結果")

        z_idx_crop = safe_slice_nav("crop_slice", cropped_volume.shape[0] - 1, label="Cropped slice")
        img_crop = apply_window(cropped_volume[z_idx_crop], wl=wl, ww=ww)
        st.image(img_crop, clamp=True, width=300)

        st.write(f"Cropped volume shape: {cropped_volume.shape}")
        st.write(
            f"Crop origin (global): "
            f"z={st.session_state['crop_origin'][0]}, "
            f"y={st.session_state['crop_origin'][1]}, "
            f"x={st.session_state['crop_origin'][2]}"
        )

        # -------------------------------------------------
        # 5. Frangi settings
        # -------------------------------------------------
        st.header("5. Frangi vessel enhancement")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            sigma_text = st.text_input("Sigmas (comma separated)", "1,2,3")
            thr_pct = st.slider("Threshold percentile", 80, 99, 90, 1)
        with col_f2:
            gamma_val = st.slider("Gamma", 1.0, 30.0, 15.0, 1.0)
            min_obj_size = st.slider("Min object size", 10, 2000, 100, 10)

        if st.button("Run Frangi on cropped ROI", use_container_width=True):
            try:
                sigmas = tuple(float(s.strip()) for s in sigma_text.split(",") if s.strip() != "")
                if len(sigmas) == 0:
                    raise ValueError("sigmas を少なくとも1つ入力してください。")

                vesselness, binary, labels, largest_component = run_frangi_3d(
                    cropped_volume=cropped_volume,
                    sigmas=sigmas,
                    gamma=gamma_val,
                    threshold_percentile=thr_pct,
                    min_object_size=min_obj_size,
                )

                centerline_points = extract_centerline_candidate_points(
                    largest_component,
                    crop_origin=st.session_state["crop_origin"],
                )

                st.session_state["vesselness"] = vesselness
                st.session_state["binary_vessel"] = binary
                st.session_state["largest_component"] = largest_component
                st.session_state["centerline_points"] = centerline_points

                st.success(f"Frangi処理完了。centerline候補点数: {len(centerline_points)}")
            except Exception as e:
                st.error(f"Frangi処理エラー: {e}")

    # -----------------------------------------------------
    # 6. Frangi result
    # -----------------------------------------------------
    vesselness = st.session_state["vesselness"]
    largest_component = st.session_state["largest_component"]
    centerline_points = st.session_state["centerline_points"]

    if vesselness is not None and largest_component is not None and cropped_volume is not None:
        st.header("6. Frangi結果表示")

        z_idx_res = safe_slice_nav("result_slice", vesselness.shape[0] - 1, label="Frangi result slice")

        base_crop_img = apply_window(cropped_volume[z_idx_res], wl=wl, ww=ww)
        vessel_img = normalize01(vesselness[z_idx_res])
        mask_img = largest_component[z_idx_res].astype(np.uint8) * 255

        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("Original cropped CT")
            st.image(base_crop_img, clamp=True, width=220)

        with col2:
            st.caption("Vesselness")
            st.image(vessel_img, clamp=True, width=220)

        with col3:
            st.caption("Largest component")
            st.image(mask_img, clamp=True, width=220)

        st.subheader("Centerline candidate overlay")

        pts_local_xy = []
        if centerline_points is not None:
            for p in centerline_points:
                if int(p["z_local"]) == z_idx_res:
                    pts_local_xy.append((p["x_local"], p["y_local"]))

        overlay_img = make_points_overlay_image(base_crop_img, pts_local_xy, figsize=(3.2, 3.2))
        st.image(overlay_img, caption="Centerline candidate point on current slice", width=300)

        if centerline_points is not None and len(centerline_points) > 0:
            st.subheader("Centerline candidate table")
            df_pts = pd.DataFrame(centerline_points)
            st.dataframe(df_pts, use_container_width=True)

            csv = df_pts.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download centerline_points.csv",
                data=csv,
                file_name="centerline_points.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()