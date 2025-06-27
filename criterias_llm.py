import numpy as np
import json
import argparse
import os
from tqdm import tqdm
from utils.metrics import MAE, MSE


def evaluate_single_sample(args):
    # --- load from JSON ---
    json_path = os.path.join(args.ckpt_base, args.ckpt_id, args.data_id, f"{str(args.date_start)}_result.json")
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return
    
    print(f"Loading data from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # --- get ground truth and predict results ---
    true_table = data.get('y_table')
    pred_table = data.get('pred')

    if true_table is None or pred_table is None:
        raise KeyError("JSON file must contain both 'y_table' (ground truth) and 'pred' (prediction) keys.")

    true_values = [float(item[1]) for item in true_table]
    pred_values = [float(item[1]) for item in pred_table]

    if len(true_values) != len(pred_values):
        raise ValueError(
            f"Data length mismatch: Ground truth has {len(true_values)} points, "
            f"while prediction has {len(pred_values)} points."
        )

    if not true_values:
        print("Warning: No data found to evaluate.")
        return

    # --- calculate metrics ---
    mae, mse = MAE(pred_values, true_values), MSE(pred_values, true_values)

    print(f"MAE for this sample: {mae:.4f}")
    print(f"MSE for this sample: {mse:.4f}")


def evaluate_all_samples(args):
    """
    Evaluate all JSON samples in the specified directory and calculate average criterias.
    """
    json_dir = os.path.join(args.ckpt_base, args.ckpt_id, args.data_id)
    if not os.path.isdir(json_dir):
        print(f"Error: Directory not found at '{json_dir}'")
        return

    # find all JSON files in the directory except the main result file
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json') and f != f"{args.data_id}_result.json"]
    
    if not json_files:
        print(f"No JSON files found in directory '{json_dir}'")
        return

    total_mae = 0.0
    total_mse = 0.0
    sample_count = 0

    # handle each JSON file iteratively
    for filename in tqdm(json_files, desc=f"Evaluating JSON files in {os.path.basename(json_dir)}", unit="file"):

        json_path = os.path.join(json_dir, filename)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # get ground truth and prediction results
            true_values = [float(item[1]) for item in data['y_table']]
            pred_values = [float(item[1]) for item in data['pred']]

            if -1 in pred_values:
                print(f"Warning: LLM has failed to predict {filename} (due to -1 in prediction values).")
                if not args.include_llm_failure:
                    continue

            if len(true_values) != len(pred_values):
                print(f"Warning: Skipping {filename} due to data length mismatch.")
                continue

            # calculate metrics
            mae, mse = MAE(pred_values, true_values), MSE(pred_values, true_values)
            
            total_mae += mae
            total_mse += mse
            sample_count += 1

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Warning: Skipping {filename} due to an error: {e}")
            continue

    # calculate average metrics
    if sample_count > 0:
        avg_mae = total_mae / sample_count
        avg_mse = total_mse / sample_count
        print(f"Average MAE for all samples: {avg_mae:.4f}")
        print(f"Average MSE for all samples: {avg_mse:.4f}")
    
    else:
        print("No valid samples were processed to calculate average metrics.")


def criterias_main(args):
    """
    Main criterias program for the script.
    """
    if args.evaluate_mode == 'single_sample':
        evaluate_single_sample(args)
    elif args.evaluate_mode == 'all_samples':
        evaluate_all_samples(args)
    else:
        raise ValueError(f"Unknown evaluate mode: {args.evaluate_mode}")
    return


if __name__ == '__main__':
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description='LLMTSF Evaluation from JSON file')
    
    # --- config ---
    parser.add_argument('--ckpt_base', type=str, default='checkpoints', help='Base directory for checkpoints')
    parser.add_argument('--ckpt_id', type=str, default='06-24-1659_gpt-4.1-nano_solar_day_ahead', help='Checkpoint folder ID')
    parser.add_argument('--data_id', type=str, default='314106', help='Data ID for evaluation')
    parser.add_argument('--date_start', type=int, default=20220203000000, help='The sample date for evaluation')
    parser.add_argument('--evaluate_mode', type=str, default='all_samples', choices=['single_sample', 'all_samples'], help='Mode to evaluate: single sample or all samples in data ID')
    parser.add_argument('--include_llm_failure', type=bool, default=False, help='Include LLM failure samples in the evaluation')

    args = parser.parse_args()
    
    criterias_main(args)
