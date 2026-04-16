from pathlib import Path

def create_habnet_config():
    project_root = Path(__file__).parent.parent
    return {
        'paths': {
            'model_dir': str(project_root / 'output' / 'models'),
            'eval_dir': str(project_root / 'output' / 'evaluation'),
            'splits_dir': str(project_root / 'data' / 'splits' / 'custom_nested_cv_20260330_124435_date_horizons1#1_more'), # label
            'datacube_dir': str(project_root / 'data' / 'datacubes_habnet_day10_2016_2024'), # input
            # 'test_data_path': str(project_root / 'data' / 'datacubes_habnet_day10_2016_2024' / 'hab_sample_with_datacube.csv') # to test simple fold
            # 'datacube_dir': str(project_root / 'data' / 'datacubes_habnet_ie04_day10_2020'), # input
            # 'test_data_path': str(project_root / 'data' / 'datacubes_habnet_ie04_day10_2020' / 'hab_sample_with_datacube.csv')
            # 'datacube_dir': str(project_root / 'data' / 'datacubes_habnet_ie04_day10_2020_allspecies'),  # input
            # 'test_data_path': str(
            #     project_root / 'data' / 'datacubes_habnet_ie04_day10_2020_allspecies' / 'hab_sample_with_datacube.csv')
        },
        'models': {
            'HABNetNas': {
                'in_channels': 10,  # needed to set up CNN model
                'backbone': 'NASNetMobile',
                'lstm_hidden': 256,
                'lstm_layers': 2,
                'bidirectional': False,
                'dropout': 0.5,
                'num_classes': 2,
                'img_size': [224, 224],
                # Add time selection parameter
                # 'time_steps': 8
            },
        },
        'training': {
            'epochs': 50,
            'batch_size': 16,
            'learning_rate': 1e-5,
            'weight_decay': 0, # 'L1 and L2 regularization did not improve performance'
            'patience': 5, # for early stopping, f1 improve
            'save_interval': 10, # for checkpoint saving
            'log_interval': 30, # For each epoch, Batch[70/105] Loss: 0.0184
            'scheduler_patience': 5,
            'scheduler_factor': 0.5,
            'use_amp': True
        },
        'split': {
            # multifold setup to load existing splits
            'use_multi_fold': False,
            'use_nested_cv': True,
        },
        ### for different horizon prediction
        'label': {
            'label_col': ['label', 'label_2', 'label_3', 'label_4', 'label_5', 'label_6', 'label_7', 'label_8', 'label_9', 'label_10',
                          'label_11', 'label_12', 'label_13', 'label_14'],
        },
        #### use_nested_cv
        'hyperparameter_tuning': {
            'search_space': {
                'HABNetNas': {
                    # 'lstm_hidden': [64, 128, 256],
                    'dropout': [0.5],
                    'learning_rate': [1e-5],
                    'batch_size': [16]
                },
            },
            'search_method': 'grid_search',  # or 'random_search'
            'n_trials': 10  # for random search
        },
        #### XAI
        'xai': {
            'enabled': False,
            'n_samples': 12,
            'captum_steps': 50,  # Integrated Gradients步数 TODO
            'save_plots': True,  # TODO
            'interactive_dashboard': True,  # TODO
            'output_dir': str(project_root / 'output' / 'xai')
        },
    }