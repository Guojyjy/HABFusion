import argparse
import re

from configs.habnet_nasnet_config import create_habnet_config # Original habnet
# from configs.fusion_config import create_datacube_multimodal_config
from scripts import MultiFoldDataSplitter, DataSplitter
from src.evaluators.base_evaluator import BaseEvaluator
from src.evaluators.nested_cv_evaluator import NestedCVModelEvaluator
from src.evaluators.multifold_evaluator import MultiFoldModelEvaluator
from src.models.habnet_nasnet import HABNetNasTrainer
# from src.models.EnhancedMultiModalTrainer import EnhancedMultiModalTrainer
# from src.models.fusion_model import DatacubeMultiModalTrainer


def _backbone_to_run_name(backbone: str) -> str:
    """Sanitize backbone name for wandb/display (e.g. vit_small_patch16_224.dino -> vit_small_patch16_224_dino)."""
    return re.sub(r"[./\\]", "_", backbone).strip("_")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train HABNet with different backbones (CNN, ViT, ConvNeXt, DINO)')
    parser.add_argument('--time_steps', type=str, default=None,
                        help='Number of time steps to use (negative for last N steps)')
    parser.add_argument('--backbone', type=str, default=None,
                        help='timm backbone name (e.g. resnet50, vit_base_patch16_224, convnext_tiny, vit_small_patch16_224.dino). '
                             'If set, runs a single backbone; otherwise uses config default.')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Override wandb project (e.g. HABShield_Benchmark). Uses config if not set.')
    parser.add_argument('--eval_only', action='store_true', help='Skip training, only evaluate existing checkpoints')
    parser.add_argument('--linear_probe', action='store_true',
                        help='Freeze backbone and train classifier only (LSTM + MLP). Use with --backbone e.g. vit_base_patch14_dinov2.')
    parser.add_argument('--linear_probe_10ch', action='store_true',
                        help='Freeze backbone except patch_embed; train patch_embed + LSTM + MLP (for 10-channel input with DINOv2).')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size (e.g. 8 for DINOv2-Large full fine-tune on 40GB GPU to avoid OOM).')
    args = parser.parse_args()

    EVALUATION_ONLY = args.eval_only  # Set to True to skip training and only inference

    # load configruation from configs/
    config = create_habnet_config()

    if args.time_steps is not None:
        config['label']['label_col'] = args.time_steps

    if args.wandb_project is not None:
        config.setdefault('wandb', {})['project'] = args.wandb_project

    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
        config['hyperparameter_tuning']['search_space']['HABNet']['batch_size'] = [args.batch_size]
        if 'CNNLSTM' in config['hyperparameter_tuning']['search_space']:
            config['hyperparameter_tuning']['search_space']['CNNLSTM']['batch_size'] = [args.batch_size]

    # Single backbone run (benchmark) or default config
    if args.backbone is not None:
        config['models']['HABNet']['backbone'] = args.backbone
        run_name = f"HABNet_{_backbone_to_run_name(args.backbone)}_TemporalCV"
        if args.linear_probe_10ch:
            config['models']['HABNet']['freeze_backbone_except_patch_embed'] = True
            run_name = f"HABNet_{_backbone_to_run_name(args.backbone)}_LinearProbe10ch_TemporalCV"
        elif args.linear_probe:
            config['models']['HABNet']['freeze_backbone'] = True
            run_name = f"HABNet_{_backbone_to_run_name(args.backbone)}_LinearProbe_TemporalCV"
        models_to_train = {run_name: HABNetNasTrainer}
    else:
        models_to_train = {
            'HABNet': HABNetNasTrainer,
            # 'HABNetNas_LSTM3_TemporalCV_NoNorm': HABNetNasTrainer,
        }

    if config['split']['use_multi_fold']:
        splitter = MultiFoldDataSplitter(config)
    elif config['split']['use_nested_cv']:
        splitter = MultiFoldDataSplitter(config)
    else:
        splitter = DataSplitter(config)

    if config['split']['use_multi_fold']:
        print("🔄 Using multifold dataset split")

        # Loading existing splits or creating new ones
        fold_splits = splitter.load_multi_fold_splits(
            fold_info_path=config['paths']['splits_dir'] + '/multi_fold_info.json')

        enable_xai = config.get('xai', {}).get('enabled', False)
        evaluator = MultiFoldModelEvaluator(config)
        all_results = evaluator.evaluate_models_multifold(fold_splits, models_to_train, False, load_existing=EVALUATION_ONLY)

    elif config['split']['use_nested_cv']:
        print("🔄 Using nested cross-validation dataset split")
        fold_splits = splitter.load_nested_cv_splits(
            fold_info_path=config['paths']['splits_dir'] + '/nested_cv_info.json')
        evaluator = NestedCVModelEvaluator(config)
        all_results = evaluator.evaluate_models_nested_cv(fold_splits, models_to_train, False,
                                                          load_existing=EVALUATION_ONLY)
    else:
        print("🔄 Using only evaluation dataset")
        test = splitter.load_test(test_data_path=config['paths']['test_data_path'])
        evaluator = BaseEvaluator(config)
        model_name = list(models_to_train.keys())[0]

        all_model_results = {} # to save results
        fold_results = []
        fold_detailed_results = {}

        for i in range(1, 4):
            fold_name = f'outer_fold_{i}_outer_train'
            if EVALUATION_ONLY:
                print("Skipping training:")
                print(f"    🔄 Loading existing model: {model_name}_{fold_name}")
                trainer = evaluator._load_existing_model(trainer_class=models_to_train[model_name], model_name=model_name, fold_name=fold_name)
                if trainer is None:
                    print(f"    ❌ No existing model found for {fold_name}")
            else:
                print("Training...")
                evaluator.train_models(test, models_to_train) # todo

            print(f"    Evaluating...")
            fold_result, predictions_dict = evaluator.evaluate_model(
                trainer,
                test['test'],
                model_name=f"{model_name}_{fold_name}",
                fold_idx=i,
            )

            fold_results.append(fold_result)
            fold_detailed_results[fold_name] = fold_result

        evaluator = MultiFoldModelEvaluator(config)
        avg_results = evaluator._compute_fold_statistics(fold_results=fold_results, model_name=model_name)
        all_model_results[model_name] = {
            'average_metrics': avg_results,
            'fold_details': fold_detailed_results
        }

        evaluator._save_multifold_results(model_name=model_name, results=all_model_results[model_name])  # json
        evaluator._generate_multifold_comparison_report(all_results=all_model_results)
