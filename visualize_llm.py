import pandas as pd
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt


def plot_prediction(indate, input_data, outdate, output_data, prediction_data, data_id, date_start, img_path):
    """
    figure the prediction visualization
    """
    plt.figure(figsize=(15, 7), dpi=300)
    plt.plot(indate, input_data, label='Input History')
    plt.plot(outdate, output_data, label='Ground Truth')
    plt.plot(outdate, prediction_data, label='Prediction', linestyle='--')
    plt.title(f'Prediction Visualization for ID: {data_id}, Sample: {date_start}')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    
    img_dir = os.path.dirname(img_path)
    if img_dir and not os.path.exists(img_dir):
        os.makedirs(img_dir)
        print(f"Created directory: {img_dir}")

    plt.savefig(img_path)
    print(f"Prediction plot saved to {img_path}")


def visualize_main(args):
    """
    Main visualize program for the script.
    """
    # --- load from json ---
    json_path = os.path.join(args.ckpt_base, args.ckpt_id, args.data_id, f"{str(args.date_start)}_result.json")
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {args.json_path}")
        return
    
    print(f"Loading data from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # input_data = 'x_table', output_data = 'y_table', prediction_data = 'pred'
    x_table = data.get('x_table', [])
    y_table = data.get('y_table', [])
    pred_table = data.get('pred', [])

    if not x_table or not y_table or not pred_table:
        raise ValueError("JSON file must contain 'x_table', 'y_table', and 'pred' keys with non-empty data.")

    x_time = [item[0] for item in x_table]
    input_np = np.array([item[1] for item in x_table])

    y_time = [item[0] for item in y_table]
    output_np = np.array([item[1] for item in y_table])

    prediction_np = np.array([item[1] for item in pred_table])

    # --- Visualization ---
    print(f"--- Visualizing data for ID: {args.data_id}, Sample: {args.date_start} ---")

    indate_dt = pd.to_datetime([str(i) for i in x_time], format='%Y%m%d%H%M%S')
    outdate_dt = pd.to_datetime([str(i) for i in y_time], format='%Y%m%d%H%M%S')

    plot_prediction(indate_dt, input_np, outdate_dt, output_np, prediction_np, args.data_id, args.date_start, args.img_path)


if __name__ == '__main__':
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description='LLMTSF Visualization from JSON file')
    
    # --- config ---
    parser.add_argument('--ckpt_base', type=str, default='checkpoints', help='Base directory for checkpoints')
    parser.add_argument('--ckpt_id', type=str, default='06-24-1659_gpt-4.1-nano_solar_day_ahead', help='Checkpoint folder ID')
    parser.add_argument('--data_id', type=str, default='314106', help='Data ID to display in the plot title')
    parser.add_argument('--date_start', type=int, default=20220203000000, help='The sample date to display in the plot title')
    parser.add_argument('--img_path', type=str, default='./imgs/visualize_llm.png', help='Path to save the prediction visualization image')

    args = parser.parse_args()
    
    visualize_main(args)