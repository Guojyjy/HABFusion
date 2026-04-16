from pathlib import Path

def create_datacube_multimodal_config():
    project_root = Path(__file__).parent.parent
    return {
        'paths': {
            'model_dir': str(project_root / 'output' / 'models'),
            'eval_dir': str(project_root / 'output' / 'evaluation'),
            'splits_dir': str(project_root / 'data' / 'splits' / 'custom_nested_cv_20260330_124435_date_horizons1#1_more'),
            'datacube_dir': str(project_root / 'data' / 'stacked_datacubes/habnet_hrrr'),
            'timeseries_insitu_csv': str(project_root / 'data' / 'datacubes_bands_day10_2016_2024' / 'hab_sample_with_datacube_past10_cellcount&insitu_buffer1.csv'), # not used cuz defined in dataset.py
        },
        'models': {
            'datacube': {
                'in_channels': 16,  # datacube通道数
                'backbone': 'vit_pwee_patch16_reg1_gap_256.sbb_in1k',
                'lstm_hidden': 128,
                'lstm_layers': 1,
                'bidirectional': False,
                'dropout': 0.2,
                'img_size': [100, 100]
            },
            'FusionModal': {
                'backbone': 'vit_pwee_patch16_reg1_gap_256.sbb_in1k',
                'in_channels': 16,
            }
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,  # 时序数据建议使用较小的batch_size
            'learning_rate': 1e-5,
            # multi-head weight strategy
            # 1 'horizon_weight_strategy': 'learnable',
            # 'use_learnable_horizon_weights': True,
            # 'horizon_weight_lr_mult': 1.0,  # Learning rate multiplier for horizon weights # 0.5  # Slower learning for weights
            'horizon_weight_strategy': 'increasing', # 'increasing'
            # 'horizon_decay_rate': 0.85,
            # 3 'horizon_weight_strategy': 'curriculum_easy_first',
            # 'curriculum_start_epoch': 5,
            # 'curriculum_end_epoch': 20,
            # 'horizon_decay_rate': 0.9,
            # 4 'horizon_weight_strategy': 'uncertainty',
            # 5
            # 'horizon_weight_strategy': 'sample_count',
            'weight_decay': 1e-5,
            'patience': 5,  # for early stopping, f1 improve
            'save_interval': 10,  # for checkpoint saving
            'log_interval': 30,  # For each epoch, Batch[70/105] Loss: 0.0184
            'scheduler_patience': 5,
            'scheduler_factor': 0.5,
            'use_amp': True  # 混合精度训练
        },
        'split': {
            'use_multi_fold': False,
            'use_nested_cv': True
        },
        'label': {
            # 'label_col': ['label', 'label_4', 'label_7', 'label_10', 'label_14'],
            'label_col': ['label', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6', 'label_7', 'label_8',
                          'label_9', 'label_10',
                          'label_11', 'label_12', 'label_13', 'label_14'],
        },
        #### use_nested_cv
        'hyperparameter_tuning': {
            'search_space': {
                'FusionModal': {
                    # 'lstm_hidden': [64, 128, 256],
                    'dropout': [0.3],
                    'learning_rate': [1e-5],
                    'batch_size': [64]
                }
            },
            'search_method': 'grid_search',  # or 'random_search'
            'n_trials': 10  # for random search
        },
        #### XAI
        'xai': {
            'enabled': False,
            'n_samples': 12,
            'captum_steps': 100,
            'save_plots': True,  # TODO
            'interactive_dashboard': True,  # TODO
            'output_dir': str(project_root / 'output' / 'xai')
        },
    }