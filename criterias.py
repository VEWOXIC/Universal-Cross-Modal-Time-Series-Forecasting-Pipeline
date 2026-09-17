import torch
import pandas as pd
import numpy as np
import os
import json
import argparse
import glob
import sys
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import model_init
from data_provider.data_factory import Data_Provider
from utils.environment_ablation import (
    ShuffledEnvironmentDataset,
    ZeroEnvironmentDataset,
)
from utils.tools import dotdict
from utils.metrics import MAE, MSE


def evaluate_full_dataset(
    loader,
    model,
    config,
    device,
    indexes,
    channel_wise,
    metric_indices=None,
    description="Running tests",
):
    """
    Evaluates all samples in a dataset using a DataLoader for efficient batch processing.
    Calculates the true MSE and MAE over the entire dataset, with an option for channel-wise evaluation.
    """
    total_mse, total_mae, num_samples = 0.0, 0.0, 0
    channel_mse, channel_mae, channel_counts = None, None, None

    for i, iter_data in tqdm(enumerate(loader), total=len(loader), desc=description):
        if indexes is not None and i not in indexes:
            continue
        with torch.no_grad():
            batch_x, batch_y, _, _, x_hetero, y_hetero, _, _, _, hetero_channel = iter_data

<<<<<<< Updated upstream
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            x_hetero = x_hetero.to(device) if x_hetero is not None else None
            y_hetero = y_hetero.to(device) if y_hetero is not None else None
            hetero_channel = hetero_channel.to(device) if hetero_channel is not None else None
=======
            batch_x = torch.as_tensor(batch_x, device=device)
            batch_y = torch.as_tensor(batch_y, device=device)
>>>>>>> Stashed changes
            
            if config.task == 'TSF':
                prediction = model(x=batch_x)
            elif config.task == 'TGTSF':
<<<<<<< Updated upstream
                prediction = model(x=batch_x, historical_events=x_hetero, news=y_hetero, channel_description=hetero_channel)
=======
                y_hetero = torch.as_tensor(y_hetero, device=device)
                hetero_channel = torch.as_tensor(hetero_channel, device=device)
                prediction = model(x=batch_x, news=y_hetero, channel_description=hetero_channel)
            elif config.task == 'MTSF':
                x_hetero = torch.as_tensor(x_hetero, device=device)
                prediction = model(x=batch_x, historical_events=x_hetero)
>>>>>>> Stashed changes
            else:
                raise ValueError(f"Unsupported task: {config.task}")
            
            prediction = prediction[:, -config.output_len:, :]
            if metric_indices is not None:
                selected_indices = torch.as_tensor(
                    metric_indices, dtype=torch.long, device=device
                )
                prediction = prediction.index_select(2, selected_indices)
                batch_y = batch_y.index_select(2, selected_indices)

            if channel_wise:
                if channel_mse is None:
                    C = prediction.shape[2]
                    channel_mse = [0.0] * C
                    channel_mae = [0.0] * C
                    channel_counts = [0] * C
                for k in range(prediction.shape[2]):
                    mse_loss = torch.nn.MSELoss()(prediction[:, :, k], batch_y[:, :, k])
                    mae_loss = torch.nn.L1Loss()(prediction[:, :, k], batch_y[:, :, k])
                    channel_mse[k] += mse_loss.item() * batch_y.size(0)
                    channel_mae[k] += mae_loss.item() * batch_y.size(0)
                    channel_counts[k] += batch_y.size(0)
            else:
                mse_loss = torch.nn.MSELoss()(prediction, batch_y)
                mae_loss = torch.nn.L1Loss()(prediction, batch_y)
                total_mae += mae_loss.item() * batch_y.size(0)
                total_mse += mse_loss.item() * batch_y.size(0)
                num_samples += batch_y.size(0)

    if channel_wise:
        return channel_mse, channel_mae, channel_counts
    else:
        return total_mse, total_mae, num_samples


def _make_shuffled_loader(loader, environment_indices, seed):
    """Clone an evaluation loader with dataset-level environment replacement."""

    shuffled_dataset = ShuffledEnvironmentDataset(
        loader.dataset,
        environment_indices=environment_indices,
        seed=seed,
    )
    return DataLoader(
        shuffled_dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
    )


def _make_zero_environment_loader(loader, environment_indices):
    """Clone an evaluation loader whose environment history is all zero."""

    zero_environment_dataset = ZeroEnvironmentDataset(
        loader.dataset,
        environment_indices=environment_indices,
    )
    return DataLoader(
        zero_environment_dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
    )


def _mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def _relative_degradation(shuffled, clean):
    if clean == 0.0:
        return float("nan")
    return (shuffled - clean) / clean * 100.0


def evaluate_shuffled_environment(
    loaders,
    model,
    config,
    device,
    filtered_samples,
    repeats,
    seed,
    output_path=None,
    checkpoint_path=None,
):
    """Compare clean and shuffled-environment target metrics."""

    if repeats <= 0:
        raise ValueError("shuffle_repeats must be positive")

    base_model = model.module if hasattr(model, "module") else model
    if not hasattr(base_model, "target_indices") or not hasattr(
        base_model, "environment_indices"
    ):
        raise ValueError(
            "--shuffle_environment requires a model exposing target_indices "
            "and environment_indices"
        )

    target_indices = [int(i) for i in base_model.target_indices.cpu().tolist()]
    environment_indices = [
        int(i) for i in base_model.environment_indices.cpu().tolist()
    ]
    print(
        "[Info] Shuffled-environment ablation: "
        f"targets={target_indices}, environment={environment_indices}, "
        f"repeats={repeats}, seed={seed}"
    )

    results = {
        "experiment": "shuffled_environment",
        "model": str(config.model),
        "checkpoint": checkpoint_path,
        "input_len": int(config.input_len),
        "output_len": int(config.output_len),
        "seed": int(seed),
        "repeats": int(repeats),
        "target_indices": target_indices,
        "environment_indices": environment_indices,
        "datasets": {},
    }
    clean_total_mse = 0.0
    clean_total_mae = 0.0
    clean_total_samples = 0
    shuffled_total_mse = [0.0] * repeats
    shuffled_total_mae = [0.0] * repeats
    shuffled_total_samples = [0] * repeats

    for dataset_number, (name, loader) in enumerate(loaders.items()):
        indexes = None
        if filtered_samples is not None:
            indexes = filtered_samples.get(name, [])

        print(f"\n[Info] Target-only ablation metrics for dataset: {name}")
        clean_mse_sum, clean_mae_sum, sample_count = evaluate_full_dataset(
            loader,
            model,
            config,
            device,
            indexes,
            channel_wise=False,
            metric_indices=target_indices,
            description=f"Clean {name}",
        )
        if sample_count == 0:
            print(f"-> No valid samples found in '{name}'")
            continue
        if len(loader.dataset) < 2:
            raise ValueError(
                f"Dataset '{name}' has fewer than two samples and cannot be shuffled"
            )

        clean_mse = clean_mse_sum / sample_count
        clean_mae = clean_mae_sum / sample_count
        repeat_mse = []
        repeat_mae = []
        repeat_degradation = []
        permutation_seeds = []

        for repeat in range(repeats):
            repeat_seed = int(seed + dataset_number * 1_000_003 + repeat)
            permutation_seeds.append(repeat_seed)
            shuffled_loader = _make_shuffled_loader(
                loader,
                environment_indices=environment_indices,
                seed=repeat_seed,
            )
            mse_sum, mae_sum, shuffled_count = evaluate_full_dataset(
                shuffled_loader,
                model,
                config,
                device,
                indexes,
                channel_wise=False,
                metric_indices=target_indices,
                description=f"Shuffled {name} [{repeat + 1}/{repeats}]",
            )
            if shuffled_count != sample_count:
                raise RuntimeError("Clean and shuffled sample counts do not match")
            shuffled_mse = mse_sum / shuffled_count
            shuffled_mae = mae_sum / shuffled_count
            repeat_mse.append(shuffled_mse)
            repeat_mae.append(shuffled_mae)
            repeat_degradation.append(
                _relative_degradation(shuffled_mse, clean_mse)
            )
            shuffled_total_mse[repeat] += mse_sum
            shuffled_total_mae[repeat] += mae_sum
            shuffled_total_samples[repeat] += shuffled_count

        mean_mse, std_mse = _mean_std(repeat_mse)
        mean_mae, std_mae = _mean_std(repeat_mae)
        mean_degradation, std_degradation = _mean_std(repeat_degradation)
        results["datasets"][str(name)] = {
            "samples": sample_count,
            "permutation_seeds": permutation_seeds,
            "clean": {"target_mse": clean_mse, "target_mae": clean_mae},
            "shuffled": {
                "target_mse": repeat_mse,
                "target_mae": repeat_mae,
                "target_mse_mean": mean_mse,
                "target_mse_std": std_mse,
                "target_mae_mean": mean_mae,
                "target_mae_std": std_mae,
                "mse_relative_degradation_percent": repeat_degradation,
                "mse_relative_degradation_mean_percent": mean_degradation,
                "mse_relative_degradation_std_percent": std_degradation,
            },
        }
        clean_total_mse += clean_mse_sum
        clean_total_mae += clean_mae_sum
        clean_total_samples += sample_count

        print(f"-> Clean target MSE/MAE: {clean_mse:.7f} / {clean_mae:.7f}")
        print(
            "-> Shuffled target MSE/MAE: "
            f"{mean_mse:.7f} +/- {std_mse:.7f} / "
            f"{mean_mae:.7f} +/- {std_mae:.7f}"
        )
        print(
            "-> Target MSE relative degradation: "
            f"{mean_degradation:.3f}% +/- {std_degradation:.3f}%"
        )

    if clean_total_samples == 0:
        raise RuntimeError("No samples were processed")

    clean_overall_mse = clean_total_mse / clean_total_samples
    clean_overall_mae = clean_total_mae / clean_total_samples
    shuffled_overall_mse = [
        total / count
        for total, count in zip(shuffled_total_mse, shuffled_total_samples)
        if count > 0
    ]
    shuffled_overall_mae = [
        total / count
        for total, count in zip(shuffled_total_mae, shuffled_total_samples)
        if count > 0
    ]
    overall_degradation = [
        _relative_degradation(value, clean_overall_mse)
        for value in shuffled_overall_mse
    ]
    mean_mse, std_mse = _mean_std(shuffled_overall_mse)
    mean_mae, std_mae = _mean_std(shuffled_overall_mae)
    mean_degradation, std_degradation = _mean_std(overall_degradation)
    results["overall"] = {
        "samples": clean_total_samples,
        "clean": {
            "target_mse": clean_overall_mse,
            "target_mae": clean_overall_mae,
        },
        "shuffled": {
            "target_mse": shuffled_overall_mse,
            "target_mae": shuffled_overall_mae,
            "target_mse_mean": mean_mse,
            "target_mse_std": std_mse,
            "target_mae_mean": mean_mae,
            "target_mae_std": std_mae,
            "mse_relative_degradation_percent": overall_degradation,
            "mse_relative_degradation_mean_percent": mean_degradation,
            "mse_relative_degradation_std_percent": std_degradation,
        },
    }

    print("\n" + "=" * 64)
    print("Shuffled-environment target-only summary")
    print(
        f"Clean target MSE/MAE: {clean_overall_mse:.7f} / "
        f"{clean_overall_mae:.7f}"
    )
    print(
        f"Shuffled target MSE: {mean_mse:.7f} +/- {std_mse:.7f}"
    )
    print(
        f"Shuffled target MAE: {mean_mae:.7f} +/- {std_mae:.7f}"
    )
    print(
        "Target MSE relative degradation: "
        f"{mean_degradation:.3f}% +/- {std_degradation:.3f}%"
    )
    print("=" * 64)

    if output_path is not None:
        output_directory = os.path.dirname(output_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2, ensure_ascii=False)
        print(f"[Info] Saved ablation results to: {output_path}")

    return results


def evaluate_zero_environment(
    loaders,
    model,
    config,
    device,
    filtered_samples,
    output_path=None,
    checkpoint_path=None,
):
    """Compare clean and zero-environment target metrics."""

    base_model = model.module if hasattr(model, "module") else model
    if not hasattr(base_model, "target_indices") or not hasattr(
        base_model, "environment_indices"
    ):
        raise ValueError(
            "--zero_environment requires a model exposing target_indices "
            "and environment_indices"
        )

    target_indices = [int(i) for i in base_model.target_indices.cpu().tolist()]
    environment_indices = [
        int(i) for i in base_model.environment_indices.cpu().tolist()
    ]
    print(
        "[Info] Zero-environment ablation: "
        f"targets={target_indices}, environment={environment_indices}"
    )

    results = {
        "experiment": "zero_environment",
        "model": str(config.model),
        "checkpoint": checkpoint_path,
        "input_len": int(config.input_len),
        "output_len": int(config.output_len),
        "target_indices": target_indices,
        "environment_indices": environment_indices,
        "zero_value": 0.0,
        "datasets": {},
    }
    clean_total_mse = 0.0
    clean_total_mae = 0.0
    zero_total_mse = 0.0
    zero_total_mae = 0.0
    total_samples = 0

    for name, loader in loaders.items():
        indexes = None
        if filtered_samples is not None:
            indexes = filtered_samples.get(name, [])

        print(f"\n[Info] Target-only zero ablation metrics for dataset: {name}")
        clean_mse_sum, clean_mae_sum, sample_count = evaluate_full_dataset(
            loader,
            model,
            config,
            device,
            indexes,
            channel_wise=False,
            metric_indices=target_indices,
            description=f"Clean {name}",
        )
        if sample_count == 0:
            print(f"-> No valid samples found in '{name}'")
            continue

        zero_loader = _make_zero_environment_loader(
            loader,
            environment_indices=environment_indices,
        )
        zero_mse_sum, zero_mae_sum, zero_sample_count = evaluate_full_dataset(
            zero_loader,
            model,
            config,
            device,
            indexes,
            channel_wise=False,
            metric_indices=target_indices,
            description=f"Zero environment {name}",
        )
        if zero_sample_count != sample_count:
            raise RuntimeError("Clean and zero-environment sample counts do not match")

        clean_mse = clean_mse_sum / sample_count
        clean_mae = clean_mae_sum / sample_count
        zero_mse = zero_mse_sum / sample_count
        zero_mae = zero_mae_sum / sample_count
        mse_degradation = _relative_degradation(zero_mse, clean_mse)
        mae_degradation = _relative_degradation(zero_mae, clean_mae)
        results["datasets"][str(name)] = {
            "samples": sample_count,
            "clean": {"target_mse": clean_mse, "target_mae": clean_mae},
            "zero_environment": {
                "target_mse": zero_mse,
                "target_mae": zero_mae,
                "mse_relative_degradation_percent": mse_degradation,
                "mae_relative_degradation_percent": mae_degradation,
            },
        }

        clean_total_mse += clean_mse_sum
        clean_total_mae += clean_mae_sum
        zero_total_mse += zero_mse_sum
        zero_total_mae += zero_mae_sum
        total_samples += sample_count

        print(f"-> Clean target MSE/MAE: {clean_mse:.7f} / {clean_mae:.7f}")
        print(f"-> Zero-env target MSE/MAE: {zero_mse:.7f} / {zero_mae:.7f}")
        print(f"-> Target MSE relative degradation: {mse_degradation:.3f}%")

    if total_samples == 0:
        raise RuntimeError("No samples were processed")

    clean_overall_mse = clean_total_mse / total_samples
    clean_overall_mae = clean_total_mae / total_samples
    zero_overall_mse = zero_total_mse / total_samples
    zero_overall_mae = zero_total_mae / total_samples
    mse_degradation = _relative_degradation(
        zero_overall_mse, clean_overall_mse
    )
    mae_degradation = _relative_degradation(
        zero_overall_mae, clean_overall_mae
    )
    results["overall"] = {
        "samples": total_samples,
        "clean": {
            "target_mse": clean_overall_mse,
            "target_mae": clean_overall_mae,
        },
        "zero_environment": {
            "target_mse": zero_overall_mse,
            "target_mae": zero_overall_mae,
            "mse_relative_degradation_percent": mse_degradation,
            "mae_relative_degradation_percent": mae_degradation,
        },
    }

    print("\n" + "=" * 64)
    print("Zero-environment target-only summary")
    print(
        f"Clean target MSE/MAE: {clean_overall_mse:.7f} / "
        f"{clean_overall_mae:.7f}"
    )
    print(
        f"Zero-env target MSE/MAE: {zero_overall_mse:.7f} / "
        f"{zero_overall_mae:.7f}"
    )
    print(f"Target MSE relative degradation: {mse_degradation:.3f}%")
    print("=" * 64)

    if output_path is not None:
        output_directory = os.path.dirname(output_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2, ensure_ascii=False)
        print(f"[Info] Saved ablation results to: {output_path}")

    return results


def main():
    """
    Main entry point for the evaluation script.
    """
    parser = argparse.ArgumentParser(description='Time Series Forecasting Model Evaluation')
    
    # --- Checkpoint and Model Config ---
    parser.add_argument('--model', type=str, default="DLinear", help="Model name (e.g., 'DLinear', 'PatchTST')")
    parser.add_argument('--data', type=str, default="ETTm1", help="Dataset name used for training (e.g., 'ETTm1')")
    parser.add_argument('--version', type=str, default="oldest", help="Model version (e.g., 'latest' 'oldest' or a specific date like '2023-10-26')")
    parser.add_argument('--input_len', type=int, default=360, help="Input sequence length")
    parser.add_argument('--output_len', type=int, default=24, help="Output sequence length (prediction horizon)")
    parser.add_argument('--checkpoint_base', type=str, default='./checkpoints/', help="Base directory for checkpoints")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for testing")
    parser.add_argument('--data_config', type=str, default=None, help="Path to the data configuration YAML file (optional)")
    parser.add_argument('--task', type=str, default="TSF", choices=["TSF", "TGTSF"], help="Task type: Time Series Forecasting or Text-Grounded TSF")
    parser.add_argument('--filtered_samples', type=str, default=None, help='Path to a JSON file containing filtered sample indexes for evaluation')
    parser.add_argument('--device', type=str, default="0", help="Device to run the model on")
    parser.add_argument('--channel_wise', type=bool, default=False, help='Channel wise testing')
    parser.add_argument(
        '--shuffle_environment',
        action='store_true',
        help=(
            'Run a target-only ablation that replaces every sample environment '
            'history with another test sample environment history'
        ),
    )
    parser.add_argument(
        '--shuffle_repeats',
        type=int,
        default=5,
        help='Number of dataset-level environment permutations',
    )
    parser.add_argument(
        '--shuffle_seed',
        type=int,
        default=2026,
        help='Base random seed for environment permutations',
    )
    parser.add_argument(
        '--ablation_output',
        type=str,
        default=None,
        help='Optional JSON path for environment-ablation results',
    )
    parser.add_argument(
        '--zero_environment',
        action='store_true',
        help=(
            'Run a target-only ablation that replaces all normalized '
            'environment history values with zero'
        ),
    )
    
    args = parser.parse_args()
    if args.shuffle_environment and args.zero_environment:
        parser.error(
            '--shuffle_environment and --zero_environment are mutually exclusive'
        )

    # --- Find and Load Checkpoint ---
    ckpt_pattern = f'_{args.model}_{args.data}_{args.output_len}_{args.input_len}'
    
    if args.version == 'latest':
        ckpt_paths = [os.path.join(args.checkpoint_base, d) for d in os.listdir(args.checkpoint_base) if ckpt_pattern in d]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found with pattern: *{ckpt_pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
    elif args.version == 'oldest':
        ckpt_paths = [os.path.join(args.checkpoint_base, d) for d in os.listdir(args.checkpoint_base) if ckpt_pattern in d]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found with pattern: *{ckpt_pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[0]
    else:
        pattern = os.path.join(args.checkpoint_base, args.version + ckpt_pattern)
        ckpt_paths = glob.glob(pattern)
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found for pattern: {pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]

    print(f"[Info] Using checkpoint path: {ckpt_path}")

    # --- Load Configuration from Checkpoint Folder ---
    config_path = os.path.join(ckpt_path, 'args.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file 'args.json' not found in {ckpt_path}")
    
    config = dotdict(json.load(open(config_path)))
    config.model_config = dotdict(config.model_config)
    config.data_config = dotdict(config.data_config) if args.data_config is None else dotdict(yaml.safe_load(open(args.data_config, 'r')))
    
    config.gpu = args.device
    config.num_workers = 0
    config.task = args.task
    config.batch_size = 1 if args.filtered_samples is not None else args.batch_size
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
        print("[Warning] CUDA is not available, use CPU instead.")
    print(f"[Info] Running on device: {device}")

    # --- Initialize and Load Model ---
    model = model_init(config.model, config.model_config, config).to(device)
    
    ckpt_file = glob.glob(os.path.join(ckpt_path, 'checkpoint*'))
    if not ckpt_file:
        raise FileNotFoundError(f"No checkpoint file (e.g., 'checkpoint.pth') found in {ckpt_path}")
    
    ckpt_file_path = ckpt_file[0]
    print(f"[Info] Loading model from: {ckpt_file_path}")
    checkpoint = torch.load(ckpt_file_path, map_location=device)

    if ckpt_file_path.endswith('.ckpt') and 'state_dict' in checkpoint:
        state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    print(f'[Info] Successfully loaded model: {config.model}')

    # --- Load Data ---
    data_provider = Data_Provider(config)
    fullloader = data_provider.get_test("loader")

    # --- Run Evaluation ---
    if args.channel_wise:
        all_mae = {}
        all_mse = {}
        all_sample_num = {}
    else:
        all_mae = 0.0
        all_mse = 0.0
        all_sample_num = 0

    filtered_samples = None
    if args.filtered_samples is not None:
        filtered_samples = json.load(open(args.filtered_samples))
        print(f"[Info] Using filtered samples from: {args.filtered_samples}")

    if args.shuffle_environment:
        evaluate_shuffled_environment(
            fullloader,
            model,
            config,
            device,
            filtered_samples=filtered_samples,
            repeats=args.shuffle_repeats,
            seed=args.shuffle_seed,
            output_path=args.ablation_output,
            checkpoint_path=ckpt_path,
        )
        return

    if args.zero_environment:
        evaluate_zero_environment(
            fullloader,
            model,
            config,
            device,
            filtered_samples=filtered_samples,
            output_path=args.ablation_output,
            checkpoint_path=ckpt_path,
        )
        return
    
    for name, loader in fullloader.items():
        print(f"\n[Info] Testing on dataset: {name}")

        if args.filtered_samples is not None:
            indexes = filtered_samples.get(name, []) # Use .get for safety
            print(f"[Info] Using {len(indexes)} filtered samples for testing.")
            print(f"[Info] Sample indexes: {indexes}")
        else:
            indexes = None
            print("[Info] Using all samples for testing.")
        
        result = evaluate_full_dataset(loader, model, config, device, indexes, args.channel_wise)

        if args.channel_wise:
            channel_mse, channel_mae, channel_counts = result
            if channel_mse is None or sum(channel_counts) == 0:
                print(f"-> No valid samples found in '{name}'")
            else:
                all_mse[name] = channel_mse
                all_mae[name] = channel_mae
                all_sample_num[name] = channel_counts
                avg_ch_mse = [m / count if count > 0 else 0 for m, count in zip(channel_mse, channel_counts)]
                avg_ch_mae = [m / count if count > 0 else 0 for m, count in zip(channel_mae, channel_counts)]
                print(f"-> Results for '{name}': Channel-wise MSE = {avg_ch_mse}, Channel-wise MAE = {avg_ch_mae}")
                print(f"-> Results for '{name}': Overall Channel MSE = {sum(avg_ch_mse) / len(avg_ch_mse):.7f}, Overall Channel MAE = {sum(avg_ch_mae) / len(avg_ch_mae):.7f}")
        else:
            total_mse, total_mae, num_samples = result
            if num_samples > 0:
                avg_mse = total_mse / num_samples
                avg_mae = total_mae / num_samples
                print(f"-> Results for '{name}': MSE = {avg_mse:.7f}, MAE = {avg_mae:.7f}")

                all_mse += total_mse
                all_mae += total_mae
                all_sample_num += num_samples
            else:
                print(f"-> No valid samples found in '{name}'")

    print("\n" + "="*50)
    print(" " * 15 + "Overall Test Summary")

    if args.channel_wise:
        # Check if there are any results to summarize
        if not all_mse:
            print("-> No results to summarize.")
        else:
            sum_mse = [sum(m) for m in zip(*all_mse.values())]
            sum_mae = [sum(m) for m in zip(*all_mae.values())]
            sum_counts = [sum(c) for c in zip(*all_sample_num.values())]
            overall_mse_list = [m / c if c > 0 else 0 for m, c in zip(sum_mse, sum_counts)]
            overall_mae_list = [m / c if c > 0 else 0 for m, c in zip(sum_mae, sum_counts)]
            print(f"-> Overall Results (All Subsets): Channel-wise MSE = {overall_mse_list}, Channel-wise MAE = {overall_mae_list}")
            overall_mse = sum(overall_mse_list) / len(overall_mse_list) if overall_mse_list else 0
            overall_mae = sum(overall_mae_list) / len(overall_mae_list) if overall_mae_list else 0
            print(f"-> Overall Results (All Subsets): MSE = {overall_mse:.7f}, MAE = {overall_mae:.7f}")

    else:
        if all_sample_num > 0:
            print(f"-> Overall Results (All Subsets): MSE = {all_mse / all_sample_num:.7f}, MAE = {all_mae / all_sample_num:.7f}")
        else:
            print("-> No samples were processed.")
    
    print("="*50)


if __name__ == '__main__':
    main()
    sys.exit(0)
