import numpy as np
import json
import argparse
import os
import sys
from tqdm import tqdm
# from utils.metrics import MAE, MSE # Assuming MAE/MSE are defined locally or in a utils file


def MAE(pred, true):
    return np.mean(np.abs(np.array(pred) - np.array(true)))

def MSE(pred, true):
    return np.mean((np.array(pred) - np.array(true)) ** 2)

# --- DATASET_STATS for NYC_traffic_speed ---
# DATASET_STATS = {
#     '204': {'mean': 42.99749363, 'var': 259.29034097},
#     '184': {'mean': 44.63174101, 'var': 168.51806643},
#     '217': {'mean': 30.20539933, 'var': 177.40726799},
#     '221': {'mean': 32.74249293, 'var': 198.34979118},
#     '149': {'mean': 25.56417156, 'var': 176.77523147},
#     '223': {'mean': 26.59874704, 'var': 167.6267469},
#     '145': {'mean': 26.35843625, 'var': 122.74204332},
#     '208': {'mean': 45.74908185, 'var': 229.15936149},
#     '207': {'mean': 45.64112676, 'var': 207.25559719},
#     '332': {'mean': 47.63285114, 'var': 199.35890278},
#     '169': {'mean': 43.10455805, 'var': 267.8900482},
#     '170': {'mean': 42.49969081, 'var': 234.73145786},
#     '171': {'mean': 19.75017172, 'var': 167.33566578},
#     '315': {'mean': 40.729702, 'var': 305.38794253},
#     '330': {'mean': 27.4390248, 'var': 159.95323266},
#     '345': {'mean': 20.27765569, 'var': 226.77337497},
#     '344': {'mean': 23.71390838, 'var': 265.45911697},
#     '191': {'mean': 24.96555085, 'var': 422.93988049},
#     '213': {'mean': 30.71104508, 'var': 111.55719426},
#     '399': {'mean': 27.45966737, 'var': 140.23724269},
#     '395': {'mean': 35.97205872, 'var': 198.60436406},
#     '410': {'mean': 46.13677064, 'var': 215.16207795},
#     '347': {'mean': 38.23007211, 'var': 167.94021133},
#     '140': {'mean': 29.37716041, 'var': 216.408017},
#     '398': {'mean': 34.91630878, 'var': 254.55783203},
#     '202': {'mean': 43.27754155, 'var': 225.37790522},
#     '168': {'mean': 39.57697463, 'var': 236.16349448},
#     '298': {'mean': 37.83218791, 'var': 277.07481937},
#     '451': {'mean': 39.44608708, 'var': 220.0128161},
#     '416': {'mean': 47.5624561, 'var': 183.51755408},
#     '190': {'mean': 24.78186664, 'var': 394.47254571},
#     '212': {'mean': 45.50875779, 'var': 144.28780229},
#     '318': {'mean': 36.01330536, 'var': 326.53812952},
#     '211': {'mean': 50.05749195, 'var': 278.0976071},
#     '319': {'mean': 43.59433998, 'var': 179.19716926},
#     '311': {'mean': 40.11583552, 'var': 342.11248314},
#     '419': {'mean': 39.25046627, 'var': 114.65554389},
#     '418': {'mean': 38.65616189, 'var': 133.78870516},
#     '387': {'mean': 48.02321567, 'var': 214.80007702},
#     '385': {'mean': 42.98978139, 'var': 232.52628978},
#     '390': {'mean': 51.42239485, 'var': 318.61089792},
#     '388': {'mean': 51.26034219, 'var': 183.87828633},
#     '412': {'mean': 42.72985692, 'var': 165.68896703},
#     '413': {'mean': 40.87038634, 'var': 171.42839455},
#     '258': {'mean': 36.10603661, 'var': 326.30397905},
#     '259': {'mean': 23.61324275, 'var': 254.84401955},
#     '394': {'mean': 31.80317398, 'var': 323.26630316},
#     '261': {'mean': 40.10679727, 'var': 275.98277387},
#     '154': {'mean': 26.21945464, 'var': 332.13235522},
#     '155': {'mean': 24.97861939, 'var': 384.83315273},
#     '153': {'mean': 33.39007106, 'var': 368.16172955},
#     '453': {'mean': 41.77528011, 'var': 206.62384694},
#     '428': {'mean': 41.78041795, 'var': 206.61024844},
#     '129': {'mean': 52.58489542, 'var': 136.59348665},
#     '126': {'mean': 48.66695563, 'var': 143.12120452},
#     '295': {'mean': 51.17509572, 'var': 334.94352465},
#     '427': {'mean': 31.5116002, 'var': 244.34152967},
#     '142': {'mean': 44.61143281, 'var': 197.3886022},
#     '185': {'mean': 34.7417432, 'var': 423.37774147},
#     '426': {'mean': 36.78335414, 'var': 217.09599078},
#     '425': {'mean': 34.72080914, 'var': 205.26330898},
#     '178': {'mean': 43.01217126, 'var': 171.95054277},
#     '422': {'mean': 34.78991743, 'var': 233.65567227},
#     '423': {'mean': 42.19292472, 'var': 218.30151475},
#     '424': {'mean': 40.90428717, 'var': 237.60003633},
#     '165': {'mean': 45.93401767, 'var': 273.95193005},
#     '177': {'mean': 30.41488495, 'var': 366.69983252},
#     '172': {'mean': 28.63796284, 'var': 328.12050187},
#     '167': {'mean': 44.04662162, 'var': 214.68352879},
#     '257': {'mean': 28.38238447, 'var': 316.37220704},
#     '222': {'mean': 36.58524283, 'var': 192.40835506},
#     '215': {'mean': 28.55406939, 'var': 206.48947569},
#     '265': {'mean': 34.70833044, 'var': 437.31761985},
#     '380': {'mean': 40.46474402, 'var': 180.06069518},
#     '137': {'mean': 43.6384669, 'var': 263.50637342},
#     '124': {'mean': 28.17228646, 'var': 250.99422301},
#     '205': {'mean': 46.12218576, 'var': 273.42113888},
#     '206': {'mean': 44.00008134, 'var': 219.86448267},
#     '405': {'mean': 43.46898087, 'var': 255.75434971},
#     '376': {'mean': 47.81120173, 'var': 200.42226514},
#     '375': {'mean': 49.56535344, 'var': 209.68935438},
#     '381': {'mean': 46.82104298, 'var': 150.10464859},
#     '377': {'mean': 46.53776909, 'var': 266.09663162},
#     '378': {'mean': 46.29662009, 'var': 260.31292674},
#     '435': {'mean': 52.37760854, 'var': 175.40035918},
#     '384': {'mean': 52.81939248, 'var': 223.35392817},
#     '157': {'mean': 29.58603907, 'var': 235.70134037},
# }

# DATASET_STATS = {
#     '314106': {'mean': 2.08970206,  'var': 14.46352061},
#     '319086': {'mean': 46.01997771, 'var': 6634.85142276},
#     '164440': {'mean': 17.250896,   'var': 1009.71119804},
#     '355827': {'mean': 29.2824691,  'var': 3087.56297112},
#     '331901': {'mean': 13.37964598, 'var': 572.21388739},
#     '332785': {'mean': 1.35890028,  'var': 5.4540398},
#     '577650': {'mean': 77.06716315, 'var': 17869.36369284},    
#     '551172': {'mean': 3.54883345,  'var': 42.85220041}, 
#     '570079': {'mean': 4.36819685,  'var': 64.22731373},
# }

# DATASET_STATS = {
#     'solar_50Hertz': {'mean': 474.09323914,  'var': 555148.90192805},
#     'solar_Amprion': {'mean': 866.75514288, 'var': 1811393.28566811},
#     'solar_TenneT': {'mean': 1358.56019045,   'var': 4424793.92567368},
#     'solar_TransnetBW': {'mean': 564.70077068,  'var': 788131.01029578},
#     'wind_50Hertz': {'mean': 182.23082484, 'var': 35782.46808419},
#     'wind_Amprion': {'mean': 108.81919813,  'var': 13440.36507843},
#     'wind_TenneT': {'mean': 201.46357377, 'var': 46407.25980962},    
#     'wind_TransnetBW': {'mean': 31.98529265,  'var': 1365.4048626}, 
# }

# DATASET_STATS = {
#     'weather_large':{
#         'mean': [9.89729964e+02,1.05105673e+01,2.84513977e+02,5.56310062e+00,7.41164607e+01,1.44013084e+01,
#                  9.82390111e+00,4.57733125e+00,6.20389295e+00,9.93009997e+00,1.21190556e+03,1.57992205e+00,
#                  2.97453814e+00,1.75514903e+02,8.58460360e-03,2.88046971e+01,1.27881363e+02,2.51100807e+02,
#                  3.01843059e+02,2.15199568e+01,4.04833509e+02], 
#         'var': [7.44130599e+01,6.38267810e+01,6.56093096e+01,3.84393318e+01,2.91928157e+02,6.42623600e+01,
#                      1.68193913e+01,2.89672749e+01,6.78861911e+00,1.72517932e+01,1.45240686e+03,5.16916307e+03,
#                      5.44593554e+03,7.22998111e+03,1.12481171e-02,1.36635686e+04,4.49693974e+04,1.70782272e+05,
#                      2.91704941e+05,6.51885916e+01,1.58052545e+05]
#     }
# }

# DATASET_STATS = {
#     'all_except_battery': {
#     'mean': [1.17661451e+02,1.47705955e+02,3.61306870e+03,1.13418380e+01,2.85834148e+03,8.61216984e+00,
#              2.54020158e+04,2.52774782e+04,2.52524909e+04,3.37944716e+03,1.92252183e+03,8.99288731e+02,
#              3.25142704e+02,2.18082520e+02,3.19128469e+02,1.42867457e+01,1.94793627e+03,7.99909330e+03,
#              2.23964005e+03,6.31853355e+03],
#     'var': [1.35550233e+02,6.15198280e+02,3.39848195e+06,1.47499634e+01,1.20899769e+06,1.61290550e+00,
#                  2.67638218e+07,2.57462673e+07,2.62461473e+07,1.69344647e+07,1.55849004e+06,1.50415509e+04,
#                  2.50268733e+03,3.17589640e+02,6.65283801e+05,2.25856297e+01,3.10426317e+05,1.60462078e+07,
#                  1.51461226e+06,5.97243838e+06]
#  }
# }

DATASET_STATS = {
    '104': {'mean': [74.87043802, 81.47070405, 42.7860227], 'var': [2.76928869, 2128.59212413, 59.83707021]},
    '107': {'mean': [74.31539696, 124.50463708, 42.7860227], 'var': [1.17079854e+00, 1.57862936e+04, 5.98370702e+01]},
    '108': {'mean': [74.31539696, 75.91645883, 42.7860227], 'var': [1.17079854e+00, 1.28853223e+04, 5.98370702e+01]},
    '120': {'mean': [70.28496277, 161.37595028, 42.7860227], 'var': [1.51343986e+00, 2.47349846e+04, 5.98370702e+01]},
    '122': {'mean': [73.7861576, 249.87310875, 42.7860227], 'var': [4.15796459e+00, 6.64676216e+04, 5.98370702e+01]},
    '208': {'mean': [71.46610966, 8.82951875, 42.7860227], 'var': [3.46123878, 745.36065024, 59.83707021]},
    '216': {'mean': [72.57323202, 145.18048473, 42.7860227], 'var': [6.26748421e+00, 2.22762181e+04, 5.98370702e+01]},
    '229': {'mean': [75.22957152, 246.01946404, 42.7860227], 'var': [2.71037640e+00, 3.68689867e+04, 5.98370702e+01]},
    '248': {'mean': [72.66964318, 110.56915087, 42.7860227], 'var': [9.73763059e+00, 1.87081697e+04, 5.98370702e+01]},
    '252': {'mean': [75.41496926, 37.59153, 42.7860227], 'var': [1.48259438e+00, 3.40354917e+03, 5.98370702e+01]},
    '261': {'mean': [74.08142949, 12.42677624, 42.7860227], 'var': [1.50704897, 56.44477819, 59.83707021]},
    '281': {'mean': [71.91093504, 38.5409267, 42.7860227], 'var': [9.48311699e-01, 1.22043381e+03, 5.98370702e+01]},
    '288': {'mean': [71.06386965, 38.5409267, 42.7860227], 'var': [2.12747091, 1220.43380655, 59.83707021]},
    '304': {'mean': [72.96783179, 289.85388525, 42.7860227], 'var': [6.00444747e+00, 7.89749854e+04, 5.98370702e+01]},
    '317': {'mean': [70.33991381, 107.19880949, 42.7860227], 'var': [5.62239624, 1786.52221793, 59.83707021]},
    '329': {'mean': [73.67290126, 117.87262247, 42.7860227], 'var': [6.49086375e+00, 1.19087724e+04, 5.98370702e+01]},
    '345': {'mean': [72.7750941, 64.07975287, 42.7860227], 'var': [1.39009785e+00, 3.91903523e+03, 5.98370702e+01]},
    '348': {'mean': [72.50058047, 149.52919126, 42.7860227], 'var': [3.91201326e+00, 1.96574734e+04, 5.98370702e+01]},
    '362': {'mean': [72.10505718, 141.86288298, 42.7860227], 'var': [3.58131836e+00, 2.37763293e+04, 5.98370702e+01]},
    '380': {'mean': [71.69537488, 42.43499561, 42.7860227], 'var': [2.47308191, 1614.30548851, 59.83707021]},
    '387': {'mean': [71.54038859, 244.56338267, 42.7860227], 'var': [2.62910630e+00, 6.37910585e+04, 5.98370702e+01]},
    '403': {'mean': [75.75371487, 214.78114009, 42.7860227], 'var': [5.09065605e+00, 2.54385768e+04, 5.98370702e+01]},
    '417': {'mean': [74.41427583, 76.28607982, 42.7860227], 'var': [2.69331307e+00, 1.69366971e+04, 5.98370702e+01]},
    '428': {'mean': [73.43179572, 174.81978503, 42.7860227], 'var': [3.54094101e+00, 2.87438421e+04, 5.98370702e+01]},
    '445': {'mean': [72.5951395, 107.44854391, 42.7860227], 'var': [9.61684072e-01, 1.04593173e+04, 5.98370702e+01]},
    '450': {'mean': [73.5846345, 74.92730915, 42.7860227], 'var': [2.59388493e+00, 4.03052537e+03, 5.98370702e+01]},
    '461': {'mean': [73.45090002, 89.47326018, 42.7860227], 'var': [2.47988977e+00, 7.19054268e+03, 5.98370702e+01]},
    '470': {'mean': [76.5128831, 168.68677774, 42.7860227], 'var': [2.04005106e+00, 3.26171413e+04, 5.98370702e+01]},
    '484': {'mean': [71.54521512, 176.71976856, 42.7860227], 'var': [1.63660674e+00, 2.03153575e+04, 5.98370702e+01]},
}


def standardize(values, mean, var):
    """Standardize a set of numerical values using Z-score."""
    values_np = np.array(values)
    mean_np = np.array(mean)
    var_np = np.array(var)
    std_dev_np = np.sqrt(np.where(var_np < 1e-6, np.inf, var_np))
    standardized_values = (values_np - mean_np) / std_dev_np
    return standardized_values.tolist()


def evaluate_all_samples(ckpt_id, data_id, train_mean, train_var, include_llm_failure):
    """
    Evaluate every JSON file in the given data_id directory and return a detailed list of results for each file.
    
    Returns:
    - file_results (list): A list of dictionaries, where each dictionary contains a file's 'filename', 'mae', 'mse', 'points'.
    - mismatch (int): The number of files with format mismatches.
    """
    json_dir = os.path.join(ckpt_id, data_id)
    if not os.path.isdir(json_dir):
        return [], 0

    summary_file = f"{data_id}_result.json"
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json') and f != summary_file]
    
    if not json_files:
        return [], 0

    file_results = []
    mismatch_count = 0
    
    is_multivariate = isinstance(train_mean, list)

    for filename in json_files:
        # --- Reset counters for each file ---
        file_total_mae = 0.0
        file_total_mse = 0.0
        file_point_count = 0
        
        json_path = os.path.join(json_dir, filename)
        try:
            if os.path.getsize(json_path) == 0:
                continue
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            true_values = [item[1] for item in data['y_table']]
            pred_values = [item[1] for item in data['pred']]
            
            if len(true_values) != len(pred_values) or not true_values:
                mismatch_count += 1
                continue
                
            for true_step, pred_step in zip(true_values, pred_values):
                if is_multivariate:
                    try:
                        # convert to float
                        true_step = [float(x) for x in true_step]
                        pred_step = [float(x) for x in pred_step]
                    except (TypeError, ValueError) as e:
                        print(f"Warning: Skipping time step in file {filename} due to conversion error: {e}")
                        continue
                    
                    if not include_llm_failure and -1.0 in pred_step:
                        continue
                    if len(true_step) != len(train_mean) or len(pred_step) != len(train_mean):
                        # Dimension mismatch is considered a file-level error, skip the entire file
                        mismatch_count += 1
                        file_point_count = 0 # Invalidate this file
                        break
                else:
                    if not include_llm_failure and pred_step == -1.0:
                        continue
                    
                    # convert to float
                    true_step = [float(true_step)]
                    pred_step = [float(pred_step)]
                    
                pred_standardized = standardize(pred_step, train_mean, train_var)
                true_standardized = standardize(true_step, train_mean, train_var)
                
                # Calculate the error for this time step
                mae = MAE(pred_standardized, true_standardized)
                mse = MSE(pred_standardized, true_standardized)
                
                # Accumulate to the total error for this file
                file_total_mae += mae
                file_total_mse += mse
                file_point_count += 1
            
            # If the file was processed successfully, calculate its average metrics and add to the results list
            if file_point_count > 0:
                avg_mae_for_file = file_total_mae / file_point_count
                avg_mse_for_file = file_total_mse / file_point_count
                file_results.append({
                    'filename': filename,
                    'mae': avg_mae_for_file,
                    'mse': avg_mse_for_file,
                    'points': file_point_count
                })

        except (KeyError, IndexError, json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Skipping file {json_path} due to error: {e}")
            continue

    return file_results, mismatch_count

def evaluate_full_dataset(args):
    """
    Iterate through all IDs, evaluate all files in each directory, optionally print a detailed report based on arguments, and finally calculate the final weighted macro-average metrics.
    """
    print(f"--- Starting full evaluation for checkpoint '{args.ckpt_id}' ---")
    
    all_results = []
    skipped_ids = []
    total_mismatches = 0
    
    if 'DATASET_STATS' not in globals():
        print("Error: 'DATASET_STATS' dictionary not found or activated in the script.")
        sys.exit(1)

    data_ids_to_process = sorted(list(DATASET_STATS.keys()))

    for data_id in tqdm(data_ids_to_process, desc="Processing all data IDs"):
        stats = DATASET_STATS[data_id]
        
        file_results, mismatch = evaluate_all_samples(
            args.ckpt_id, 
            data_id, 
            stats['mean'], 
            stats['var'], 
            args.include_llm_failure
        )
        
        total_mismatches += mismatch
        
        if file_results:
            # Only print detailed report if detect_anomaly is True ---
            if args.detect_anomaly:
                print(f"\n\n" + "="*20 + f" Anomaly Detection Report for Data ID: {data_id} " + "="*20)
                sorted_files = sorted(file_results, key=lambda x: x['mse'], reverse=True)
                
                print(f"{'JSON Filename':<40} | {'Avg MAE':>12} | {'Avg MSE':>15} | {'# Points':>10}")
                print("-" * 85)
                for result in sorted_files:
                    print(f"{result['filename']:<40} | {result['mae']:>12.4f} | {result['mse']:>15.4f} | {result['points']:>10}")
            
            all_results.extend(file_results)
        else:
            skipped_ids.append(data_id)
    
    print("\n\n" + "="*28 + " Final Evaluation Summary " + "="*28)
    if all_results:
        maes = [r['mae'] for r in all_results]
        mses = [r['mse'] for r in all_results]
        weights = [r['points'] for r in all_results]

        final_mae = np.average(maes, weights=weights)
        final_mse = np.average(mses, weights=weights)
        
        print(f"Successfully processed {len(data_ids_to_process) - len(skipped_ids)} out of {len(data_ids_to_process)} data IDs.")
        print(f"A total of {len(all_results)} sample files were evaluated, containing {sum(weights)} valid time steps.")
        print("\nMetrics are weighted averages (the average error of each file is weighted by its length):")
        print(f"Overall Weighted Average MAE (based on standardized data): {final_mae:.4f}")
        print(f"Overall Weighted Average MSE (based on standardized data): {final_mse:.4f}")
        print(f"Found {total_mismatches} sample mismatches or format errors during processing.")
    else:
        print("No valid data was processed across all data IDs.")

    if skipped_ids:
        print("\n------------------------------------------------------------")
        print(f"The following {len(skipped_ids)} data IDs were skipped because no valid samples were found:")
        print("  " + ", ".join(skipped_ids))
        print("------------------------------------------------------------")
    
    print("="*72)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Evaluate LLM time series predictions from JSON files.')
    
    parser.add_argument('--ckpt_base', type=str, default='checkpoints', help='Base directory for checkpoints.')
    parser.add_argument('--model', type=str, default="qwen3-14b", help="Model name (e.g., 'Deepseek-R1', 'Qwen2.5-14B')")
    parser.add_argument('--data', type=str, default="Bear_room", help="Dataset name used for training (e.g., 'ETTm1')")
    parser.add_argument('--version', type=str, default="oldest", help="Model version (e.g., 'latest' 'oldest' or a specific date like '2023-10-26')")
    parser.add_argument('--input_len', type=int, default=288, help="Input sequence length")
    parser.add_argument('--output_len', type=int, default=12, help="Output sequence length (prediction horizon)")
    
    parser.add_argument('--include_llm_failure', action='store_true', help='Include samples where LLM prediction failed in the evaluation.')
    parser.add_argument('--detect_anomaly', action='store_true', help='Print detailed loss for each subset (JSON file) to locate outliers.')
    
    args = parser.parse_args()

    # --- Find and Load Checkpoint ---
    ckpt_pattern = f'_{args.model}_{args.data}_{args.output_len}_{args.input_len}'
    
    if args.version == 'latest':
        ckpt_paths = [os.path.join(args.ckpt_base, d) for d in os.listdir(args.ckpt_base) if ckpt_pattern in d]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found with pattern: *{ckpt_pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]
    elif args.version == 'oldest':
        ckpt_paths = [os.path.join(args.ckpt_base, d) for d in os.listdir(args.ckpt_base) if ckpt_pattern in d]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found with pattern: *{ckpt_pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[0]
    else:
        pattern = os.path.join(args.ckpt_base, args.version + ckpt_pattern)
        ckpt_paths = glob.glob(pattern)
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint found for pattern: {pattern}")
        ckpt_paths.sort()
        ckpt_path = ckpt_paths[-1]

    args.ckpt_id = ckpt_path

    evaluate_full_dataset(args)

if __name__ == '__main__':
    main()