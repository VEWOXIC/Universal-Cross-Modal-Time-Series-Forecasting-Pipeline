import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_single_parquet_by_time_range(file_path, title, start_date='2019-01-01', end_date='2020-12-31', output_filename="single_plot.pdf"):
    """
    Reads data from a single Parquet file, handles timezone, filters by time range, and plots the data.
    
    Parameters:
    file_path (str): Path to a single Parquet file.
    title (str): Title of the plot.
    start_date (str): Start date for data filtering.
    end_date (str): End date for data filtering.
    output_filename (str): Output PDF filename.
    """
    # --- Define font sizes for axis labels and ticks ---
    TITLE_FONTSIZE = 22
    LABEL_FONTSIZE = 18
    TICK_FONTSIZE = 16

    # --- Modification: Create a single subplot (1 row, 1 column) with a more suitable size for a single plot ---
    fig, ax = plt.subplots(figsize=(20, 5)) 
    
    try:
        df = pd.read_parquet(file_path)
        
        time_col_name = df.columns[0]
        # 1. Convert to pandas datetime objects uniformly
        df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce', utc=True)
        
        # 2. Handle timezone information
        if df[time_col_name].dt.tz is not None:
            print(f"Timezone detected in '{os.path.basename(file_path)}', removing...")
            df[time_col_name] = df[time_col_name].dt.tz_localize(None)
        
        df.dropna(subset=[time_col_name], inplace=True)
        df.set_index(time_col_name, inplace=True)
        
        df_filtered = df.loc[start_date:end_date]
        
        if df_filtered.empty:
            print(f"Warning: No data found between {start_date} and {end_date} in file '{os.path.basename(file_path)}'.")
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=16)
        else:
            x_data = df_filtered.index
            y_data = df_filtered.iloc[:, 0]
            ax.plot(x_data, y_data)
            ax.grid(True)
            
            # --- Set axis ticks and labels ---
            # ax.set_xlabel('Date', fontsize=LABEL_FONTSIZE)
            # ax.set_ylabel('Value', fontsize=LABEL_FONTSIZE)
            ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")
        ax.text(0.5, 0.5, 'Processing Error', ha='center', va='center', color='red', fontsize=16)
    
    # --- Modification: Set title directly ---
    # ax.set_title(title, fontsize=TITLE_FONTSIZE, loc='center')

    fig.autofmt_xdate(rotation=30, ha='right')
    plt.tight_layout(pad=1.5)
    
    fig.savefig(output_filename, bbox_inches='tight')
    print(f"Plot successfully saved as: {output_filename}")
    # plt.show()


# --- Main program entry ---
if __name__ == "__main__":
    # --- Modification: Now only one file path is needed ---
    my_file = 'data/Canada_photovoltaics_plants/time_series/id_164440.parquet'
    
    # Ensure directory exists
    dir_name = os.path.dirname(my_file)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Created directory: {dir_name}")

    # To allow the code to run, create a dummy file if it doesn't exist
    try:
        pd.read_parquet(my_file)
    except FileNotFoundError:
        print("Warning: Data file not found, creating dummy data for demonstration...")
        # Create dummy data
        dates = pd.to_datetime(pd.date_range(start='2024-11-15', end='2025-01-15', freq='H', tz='UTC'))
        data = np.random.rand(len(dates)) * 100 + 50 * np.sin(np.linspace(0, 20*np.pi, len(dates)))
        df = pd.DataFrame({'datetime': dates, 'value': data})
        df.to_parquet(my_file)
        print("Dummy data created.")

    # --- Modification: Call the new function with a single file path and title ---
    plot_single_parquet_by_time_range(
        file_path=my_file,
        title='Canada Calgary Photovoltaics',
        start_date='2024-12-01',
        end_date='2025-01-01',
        output_filename="imgs/canada_photovoltaics_plot.png"
    )