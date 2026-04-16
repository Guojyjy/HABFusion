# src/dataset.py
# PyTorch Dataset for HABNet datacubes

import os
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.utils.normalization_utils import load_fold_normalization_stats


def extract_fold_name_simple(model_name: str) -> str:
    """
    find 'outer_fold'
    """
    parts = model_name.split('_')
    for i, part in enumerate(parts):
        if part == 'outer':
            return '_'.join(parts[i:])

    raise ValueError(f"Cannot find {model_name}")


class MultiheadDataset(Dataset):
    """
    PyTorch Dataset for loading spatiotemporal datacubes for HAB classification.

    Expects an index CSV or dataframe with columns:
      - cube_path: relative path to the .npy datacube file
      - label: integer label (0 = no HAB, 1 = HAB)

    Each datacube .npy file should have shape (T, H, W, M) where:
      T = number of time steps
      H = spatial height (pixels)
      W = spatial width (pixels)
      M = number of spectral/tidal channels

    The dataset will transpose each cube to (T, M, H, W) before converting to Tensor.
    """

    def __init__(self, index_rec: str or pd.DataFrame, cubes_dir: str, label_col='label', transform=None,
                 masked_label: bool = False,
                 fold_index: int = None, fold_model_name = ''):
        """
        fold_index: for naming normalization stats file
        masked_label: whether to mask label with buffer
        """
        self.cubes_dir = cubes_dir
        self.label_col = label_col
        self.transform = transform

        # Load index file
        if isinstance(index_rec, str):
            index_rec = pd.read_csv(index_rec)

        # filter out rows with missing labels
        if isinstance(self.label_col, (list, tuple)):
            # 多个 horizon
            label_cols = list(self.label_col)
            index_rec = index_rec[index_rec[label_cols].notna().any(axis=1)]
        else:
            # 单个 label 列
            index_rec = index_rec[index_rec[self.label_col].notna()]

        # load normalization configuration
        self.modal_list = ['bathymetry',
                           'chlor_a', 'Rrs_412', 'Rrs_443', 'Rrs_488', 'Rrs_531', 'Rrs_555', 'par', 'sst',
                           'chlor_a_terra',
                           # from other data sources not in HABNet
                           'TMP', 'GUST', 'SNOWC', 'HGT', 'CRAIN', 'UGRD', 'VGRD', 'DSWRF', 'APCP', 'PRATE', 'TCDC',
                           'LCDC',
                           'SPFH', 'PRES', 'PWAT']

        self.selected_modal = ([i for i in range(10)]
                               + [self.modal_list.index('TMP')] + [self.modal_list.index('UGRD')]
                               + [self.modal_list.index('VGRD')] + [self.modal_list.index('DSWRF')]
                               + [self.modal_list.index('APCP')] + [self.modal_list.index('GUST')])

        stats_path = 'data/splits/custom_nested_cv_20260330_111631_date/modality_log_statistics.json'
        fold_name = extract_fold_name_simple(fold_model_name)
        means, stds = load_fold_normalization_stats(
            stats_path=stats_path,
            fold_name=fold_name,
            selected_modal=self.selected_modal,
            modal_list=self.modal_list,
        )
        self.mod_means = means.reshape(1, -1, 1, 1) # (T, M, H, W)
        self.mod_stds = stds.reshape(1, -1, 1, 1)
        # print(self.mod_means)

        # mask label with buffer
        # if masked_label:
        #     mask_file_path = '/data/datacubes_habnet_day10_2016_2024/hab_sample_with_datacube_horizon_labels_buffer1#1_more_mask.csv'
        #     mask_df = pd.read_csv(mask_file_path)
        #     mask_df['date'] = pd.to_datetime(mask_df['date'])
        #     # 合并索引和mask数据
        #     index_rec['date'] = pd.to_datetime(index_rec['date'])
        #     index_rec = index_rec.merge(
        #         mask_df[['lat', 'lon', 'date', f'{label_col}_mask']],
        #         on=['lat', 'lon', 'date'],
        #         how='left'
        #     )
        #
        #     # 只保留mask为1的记录
        #     index_rec = index_rec[index_rec[f'{label_col}_mask'] == 1]

        if isinstance(self.label_col, (list, tuple)):
            missing = [c for c in self.label_col if c not in index_rec.columns]
            if missing:
                raise ValueError(f"Missing label columns in index: {missing}")
        else:
            if self.label_col not in index_rec.columns and self.label_col.upper() not in index_rec.columns:
                raise ValueError(f"Index CSV must contain {self.label_col} or {self.label_col.upper()} column")

        self.index = index_rec.reset_index(drop=True)

        # load past HAB data
        past_file = 'data/datacubes_habnet_day10_2016_2024/hab_sample_with_datacube_past10_cellcount&insitu_buffer1.csv'
        self.past_hab_df = pd.read_csv(past_file) # past 10 days of HAB ground truth data: cellcount, salinity, water_temp
        self.past_hab_df.set_index(['lat', 'lon', 'date'], inplace=True)
        self.past_hab_stats = {
            'CELLCOUNT': {
                'mean': np.float64(7.194074796280797),
                'std': np.float64(6.563667194462696)},
            'SALINITY': {
                'mean': np.float64(32.64680846986682),
                'std': np.float64(4.916168641019127)},
            'WATER_TEMP': {
                'mean': np.float64(24.88862814009778),
                'std': np.float64(4.580408705869035)}
        }

        # Cache all available files to avoid repeated os.listdir calls
        self.available_files = set(os.listdir(cubes_dir))

    def __len__(self):
        # Total number of samples
        return len(self.index)

    def get_time_features(self, event_date):
        # 提取时间特征
        year_idx = event_date.year - 2016  # 假设从2016年开始
        season_idx = (event_date.month % 12) // 3  # 0: Winter, 1: Spring, 2: Summer, 3: Autumn
        day_idx = event_date.timetuple().tm_yday - 1  # 0-365

        time_features = torch.tensor([year_idx, season_idx, day_idx], dtype=torch.float32) # (3,)

        # 假设每个数据立方体包含多个时间步的数据
        # 需要为每个时间步生成对应的时间特征
        # 这里需要根据实际数据结构调整

        # 示例：如果有T个时间步，需要生成T个时间特征
        # time_features_per_timestep = []
        # for t in range(10):  # T是时间步数
        #     # 为每个时间步计算对应的时间特征
        #     current_date = event_date + pd.Timedelta(days=t-9)  # 和datacube的时间索引一致，最后是today
        #     year_idx = current_date.year - 2016
        #     season_idx = (current_date.month % 12) // 3
        #     day_idx = current_date.timetuple().tm_yday
        #     time_features_per_timestep.append([year_idx, season_idx, day_idx])
        #
        # return torch.tensor(time_features_per_timestep, dtype=torch.float32)  # (T, 3)
        return time_features  # (3,)

    def get_past_hab(self, row):
        """
            获取过去 10 天的 HAB 相关数据，包括细胞计数、盐度和水温

            Args:
                row: 数据集中的行数据，包含 lat, lon, date 等信息

            Returns:
                torch.Tensor: 形状为 (T, 3) 的张量，T=10，3 代表 CELLCOUNT、SALINITY、WATER_TEMP
            """
        # 获取当前样本的位置和日期信息
        event_lat = row['lat'] if 'lat' in row else row['LATITUDE']
        event_lon = row['lon'] if 'lon' in row else row['LONGITUDE']
        event_date = pd.to_datetime(row['date']).normalize() if 'date' in row \
            else pd.to_datetime(row['SAMPLE_DATE']).normalize()

        # 使用索引快速查找匹配的记录
        date_str = event_date.strftime('%Y-%m-%d')
        matched_data = self.past_hab_df.loc[(event_lat, event_lon, date_str)]

        # 提取过去 10 天的数据
        past_features = []
        for i in range(9, -1, -1):  # D-9 到 D-0, from past to present
            cellcount_key = f'CELLCOUNT_D-{i}'
            salinity_key = f'SALINITY_D-{i}'
            water_temp_key = f'WATER_TEMP_D-{i}'

            cellcount = matched_data[cellcount_key] if cellcount_key in matched_data else 0
            salinity = matched_data[salinity_key] if salinity_key in matched_data else 0
            water_temp = matched_data[water_temp_key] if water_temp_key in matched_data else 0

            # 处理缺失值
            cellcount = 0 if pd.isna(cellcount) else float(cellcount)
            salinity = 0 if pd.isna(salinity) else float(salinity)
            water_temp = 0 if pd.isna(water_temp) else float(water_temp)


            cellcount_log = np.log1p(np.maximum(cellcount, 0))
            cellcount_norm = (cellcount_log - self.past_hab_stats['CELLCOUNT']['mean']) / self.past_hab_stats['CELLCOUNT'][
                'std']
            salinity_norm = (salinity - self.past_hab_stats['SALINITY']['mean']) / self.past_hab_stats['SALINITY'][
                'std']
            water_temp_norm = (water_temp - self.past_hab_stats['WATER_TEMP']['mean']) / \
                              self.past_hab_stats['WATER_TEMP']['std']

            past_features.append([cellcount_norm, salinity_norm, water_temp_norm])

        return torch.tensor(past_features, dtype=torch.float32)  # (10, 3)


    def __getitem__(self, idx):
        # Read the row
        row = self.index.iloc[idx]
        event_lat = row['lat'] if 'lat' in self.index.columns else row['LATITUDE']
        event_lon = row['lon'] if 'lon' in self.index.columns else row['LONGITUDE']
        if 'date' in self.index.columns:
            event_date = pd.to_datetime(row['date'])
        elif 'SAMPLE_DATE' in self.index.columns:
            event_date = pd.to_datetime(row['SAMPLE_DATE'])
        elif 'uid' in self.index.columns:
            event_date = pd.to_datetime(row['uid'].split('_')[0])

        time_features = self.get_time_features(event_date)
        past_hab = self.get_past_hab(row)

        # cube_file = f"cube_lat{event_lat:.4f}_lon{event_lon:.4f}_date{event_date.strftime('%Y%m%d')}.npy"
        cube_file = f"stacked_lat{event_lat:.4f}_lon{event_lon:.4f}_date{event_date.strftime('%Y%m%d')}.npy"

        # Check if file exists using cached set
        if cube_file not in self.available_files:
            raise FileNotFoundError(f"Datacube file {cube_file} not found in {self.cubes_dir}")

        # Load datacube
        cube = np.load(os.path.join(self.cubes_dir, cube_file))  # (T, H, W, M)
        if cube.ndim != 4:
            raise ValueError(f"Unexpected datacube shape {cube.shape}, expected 4D array")

        # Reorder to (T, M, H, W)
        t, h, w, m = cube.shape
        cube = cube.transpose(0, 3, 1, 2) # to align with (batch, channels, height, width)

        # Make sure the cube has the expected shape TODO why shape is (1, 10, 100, 101)
        target_h, target_w = 100, 100
        if h != target_h or w != target_w:
            # Resize or crop to target dimensions
            if h > target_h or w > target_w:
                # Crop if larger
                cube = cube[:, :, :target_h, :target_w]
                print(f"Warning: datacube size is ({h}, {w}), cropped to {target_h}x{target_w}")
            else:
                # Pad if smaller
                pad_h = target_h - h
                pad_w = target_w - w
                cube = np.pad(cube, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                print(f"Warning: datacube size is ({h}, {w}), padded to {target_h}x{target_w}")

        cube = cube[:, self.selected_modal, :, :]  # (T, M, H, W)

        # Preprocess the cube
        # 1. HRRR having -999 values for missing url of certain days, not impact normalization than replace with 0
        cube = cube.astype(np.float32)
        cube[cube == -999] = np.nan
        # 2. Transformations
        ## a) log1p transform
        chl_idx = [self.modal_list.index('chlor_a'), self.modal_list.index('chlor_a_terra')]
        cube[:, chl_idx, :, :] = np.log1p(np.where(np.isnan(cube[:, chl_idx]), 0, cube[:, chl_idx]) + 1e-6)
        # from other data sources not in HABNet
        apcp_idx = self.selected_modal.index(self.modal_list.index('APCP'))
        cube[:, apcp_idx, :, :] = np.log1p(np.where(np.isnan(cube[:, apcp_idx]), 0, cube[:, apcp_idx]))
        dswrf_idx = self.selected_modal.index(self.modal_list.index('DSWRF'))
        cube[:, dswrf_idx, :, :] = np.log1p(np.where(np.isnan(cube[:, dswrf_idx]), 0, cube[:, dswrf_idx]))
        # b) clip
        rrs_indices = [self.modal_list.index(f'Rrs_{i}') for i in [412, 443, 488, 531, 555]]
        cube[:, rrs_indices, :, :] = np.clip(np.where(np.isnan(cube[:, rrs_indices]), 0, cube[:, rrs_indices]), 0, 0.05)
        # 3. Replace any -999 values with 0
        cube = np.nan_to_num(cube, nan=0.0)
        # 4. Standardize
        cube = (cube - self.mod_means) / self.mod_stds

        # Convert to torch.Tensor
        tensor_cube = torch.from_numpy(cube).float()

        if isinstance(self.label_col, (list, tuple)):
            # 例如 ['label_1', ..., 'label_14']
            label_vals = row[self.label_col].to_numpy(dtype=float)  # 可能有 NaN
            exist_mask = ~np.isnan(label_vals)  # True 表示这个 horizon 有标签

            # 将 NaN 填成 0（不会参与 loss，因为 mask=0）
            label_vals = np.nan_to_num(label_vals, nan=0.0)

            label = torch.from_numpy(label_vals.astype(np.float32))  # (H,)
            label_mask = torch.from_numpy(exist_mask.astype(np.float32))  # (H,)
        else:
            row_label = row[self.label_col]
            label = torch.tensor(int(row_label), dtype=torch.long)
            label_mask = torch.tensor([1.0], dtype=torch.float32)  # (1,)

        sample = {
            'cube': tensor_cube,
            'label': label,
            'label_mask': label_mask,
            'filename': cube_file,
            'lat': event_lat,
            'lon': event_lon,
            'date': event_date.strftime('%Y-%m-%d'),
            'time_features': time_features,
            'past_hab': past_hab
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

# Example of using the dataset:
# from src.dataset import HABDataset
# train_ds = HABDataset(index_rec='data/processed/train_index.csv', cubes_dir='output/datacubes')
# train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)

if __name__ == '__main__':
    # testing
    model_name = "CNNLSTM_vit_pwee_Temporal+6Norm_None_outer_fold_1_inner_fold_1"
    fold_name = extract_fold_name_simple(model_name)
    print(fold_name)  # outer_fold_1_inner_fold_1
    model_name = "CNNLSTM_vit_pwee_Temporal+6Norm_None_outer_fold_1_outer_train"
    fold_name = extract_fold_name_simple(model_name)
    print(fold_name)  # outer_fold_1_outer_train

    datacube_dir = '../../data/datacubes_bands_day1_2005onwards/'
    loader = MultiheadDataset(index_rec=datacube_dir + 'hab_sample_with_datacube.csv', cubes_dir=datacube_dir)
    # sample = loader[0]
    # print("Keys:", sample.keys())
    # print("Cube shape:", sample['cube'].shape)
    # print("Label:", sample['label'].item())
    # print("Filename:", sample['filename'])
    # print("Location: Lat {}, Lon {}".format(sample['lat'], sample['lon']))
    # print("Date:", sample['date'])

    shapes = []
    count_wrong = 0
    correct_indices = []
    #todo Check  torch.Size([10, M, 100, 101])
    source_dir = datacube_dir
    target_dir = datacube_dir + 'wrongShape/'
    os.makedirs(target_dir, exist_ok=True)
    for i in range(len(loader)):
        if loader[i] is not None:
            shapes.append(loader[i]['cube'].shape)
            if loader[i]['cube'].shape != (1, 10, 100, 100):
                # print(loader[i])
                count_wrong += 1
                cube_file = loader[i]['filename']
                source_path = os.path.join(source_dir, cube_file)
                target_path = os.path.join(target_dir, cube_file)
                shutil.move(source_path, target_path)
            else:
                # to save new csv for the corrected cubes
                correct_indices.append(i)

    new_index_df = loader.index.iloc[correct_indices].reset_index(drop=True)
    new_index_path = datacube_dir + 'hab_sample_with_datacube_corrected.csv'
    new_index_df.to_csv(new_index_path, index=False)

    print("Total wrong shapes:", count_wrong)
    print(set(shapes))

