import os
import numpy as np
from pathlib import Path

from src.utils import ensure_dir
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_spectral_shape(
    val_minus: np.ndarray,
    val_center: np.ndarray,
    val_plus: np.ndarray,
    lam_minus: float,
    lam_center: float,
    lam_plus: float,
) -> np.ndarray:
    """
    Compute spectral shape (SS) per pixel.

    Parameters
    ----------
    val_minus : np.ndarray
        Reflectance at lambda_minus, shape (..., H, W)
    val_center : np.ndarray
        Reflectance at lambda_center, shape (..., H, W)
    val_plus : np.ndarray
        Reflectance at lambda_plus, shape (..., H, W)
    lam_minus : float
        Left wavelength, e.g. 443
    lam_center : float
        Center wavelength, e.g. 488
    lam_plus : float
        Right wavelength, e.g. 555

    Returns
    -------
    np.ndarray
        SS map with the same spatial shape.
    """
    interp = val_minus + (val_plus - val_minus) * (
        (lam_center - lam_minus) / (lam_plus - lam_minus)
    )
    ss = val_center - interp
    return ss


def compute_roi_ss_from_cube(
    cube: np.ndarray,
    center_idx: int = 3,
    minus_idx: int = 2,
    plus_idx: int = 4,
    lam_center: float = 488.0,
    lam_minus: float = 443.0,
    lam_plus: float = 555.0,
) -> float:
    """
    Compute ROI-level SS from a datacube.

    Input cube shape: (T, H, W, M)
    The function:
      1) reorders to (T, M, H, W)
      2) computes pixel-based SS
      3) averages over the whole ROI (and time)

    Returns
    -------
    float
        Final ROI-level SS value.
    """
    if cube.ndim != 4:
        raise ValueError(f"Expected cube with 4 dims (T, H, W, M), got {cube.shape}")

    # Reorder to (T, M, H, W)
    cube = cube.transpose(0, 3, 1, 2)

    t, m, h, w = cube.shape
    if max(center_idx, minus_idx, plus_idx) >= m:
        raise IndexError(
            f"Band index out of range. Cube has M={m}, "
            f"but got minus_idx={minus_idx}, center_idx={center_idx}, plus_idx={plus_idx}"
        )

    band_minus = cube[-1, minus_idx, :, :] # The last day
    band_center = cube[-1, center_idx, :, :]
    band_plus = cube[-1, plus_idx, :, :]

    # Optional: mask invalid values if needed
    # Here I assume NaN already marks invalid values.
    ss_map = compute_spectral_shape(
        val_minus=band_minus,
        val_center=band_center,
        val_plus=band_plus,
        lam_minus=lam_minus,
        lam_center=lam_center,
        lam_plus=lam_plus,
    )

    # Average over all valid pixels and all T
    ss_roi = np.nanmean(ss_map)

    return float(ss_roi)


def process_csv_and_add_ss(
    input_csv: str,
    cube_dir: str,
    output_csv: str,
    lat_col: str = "lat",
    lon_col: str = "lon",
    date_col: str = "date",
    center_idx: int = 3,
    minus_idx: int = 2,
    plus_idx: int = 4,
    lam_center: float = 488.0,
    lam_minus: float = 443.0,
    lam_plus: float = 555.0,
) -> pd.DataFrame:
    """
    Read CSV, compute SS for each record from its matching datacube, and save new columns:
      - SS
      - SS_predict  (1 if SS < 0 else 0)

    Cube filename format:
      cube_lat{event_lat:.4f}_lon{event_lon:.4f}_date{YYYYMMDD}.npy
    """
    df = pd.read_csv(input_csv)
    df[date_col] = pd.to_datetime(df[date_col])

    ss_values = []
    ss_preds = []

    for idx, row in df.iterrows():
        event_lat = float(row[lat_col])
        event_lon = float(row[lon_col])
        event_date = row[date_col]

        cube_file = (
            f"cube_lat{event_lat:.4f}_lon{event_lon:.4f}_"
            f"date{event_date.strftime('%Y%m%d')}.npy"
        )
        cube_path = os.path.join(cube_dir, cube_file)

        if not os.path.exists(cube_path):
            print(f"[Warning] Missing cube file for row {idx}: {cube_path}")
            ss_values.append(np.nan)
            ss_preds.append(np.nan)
            continue

        try:
            cube = np.load(cube_path)

            ss = compute_roi_ss_from_cube(
                cube=cube,
                center_idx=center_idx,
                minus_idx=minus_idx,
                plus_idx=plus_idx,
                lam_center=lam_center,
                lam_minus=lam_minus,
                lam_plus=lam_plus,
            )

            ss_pred = 1 if ss < 0 else 0

            ss_values.append(ss)
            ss_preds.append(ss_pred)

        except Exception as e:
            print(f"[Error] Failed at row {idx}, file={cube_path}, error={e}")
            ss_values.append(np.nan)
            ss_preds.append(np.nan)

    df["SS"] = ss_values
    df["SS_predict"] = ss_preds

    df.to_csv(output_csv, index=False)
    print(f"Saved updated CSV to: {output_csv}")

    return df


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    input_csv = str(project_root / 'data' / 'splits' / 'custom_nested_cv_20260330_111631_date' / "outer_fold_3" / "test.csv")
    cube_dir = str(project_root / 'data' / 'datacubes_habnet_day10_2016_2024' )
    output_dir = str(project_root / 'output' / 'evaluation' / "baseline_ss_updated" / "outer_fold_3")
    ensure_dir(output_dir)
    output_csv = output_dir+"/test_output_with_ss.csv"
    print(output_csv)


    df_out = process_csv_and_add_ss(
        input_csv=input_csv,
        cube_dir=cube_dir,
        output_csv=output_csv,
        lat_col="lat",
        lon_col="lon",
        date_col="date",
        center_idx=3,   # 488 nm
        minus_idx=2,
        plus_idx=4,
        lam_center=488.0,
        lam_minus=412.0,
        lam_plus=531.0,
    )

    print(df_out.head())
    df = df_out

    # or read CSV
    # df = pd.read_csv(output_csv)

    df = df.dropna(subset=["label", "SS_predict"])

    y_true = df["label"].astype(int)
    y_pred = df["SS_predict"].astype(int)

    # metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")