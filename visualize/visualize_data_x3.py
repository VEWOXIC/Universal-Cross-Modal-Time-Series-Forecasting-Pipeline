import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_parquets_by_time_range(file_paths, start_date='2019-01-01', end_date='2020-12-31'):
    """
    Reads data from Parquet files, handles timezone differences, filters by time range, and plots them side-by-side.
    """
    if len(file_paths) != 3:
        raise ValueError("This function requires a list of three file paths.")

    # --- Modification: Define font sizes for axis labels and ticks for easy adjustment ---
    TITLE_FONTSIZE = 20
    LABEL_FONTSIZE = 16
    TICK_FONTSIZE = 14

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6)) # Slightly increased figure size to accommodate larger fonts
    
    sub_labels = ['(a)', '(b)', '(c)']
    titles = ['Canada Calgary Photovoltaics', 'Germany Renewable Energy Grid', 'NYC Traffic Speed']
    
    if len(titles) != len(file_paths):
        raise ValueError("Number of titles must match number of files.")

    for i, file_path in enumerate(file_paths):
        ax = axes[i]
        try:
            df = pd.read_parquet(file_path)
            
            time_col_name = df.columns[0]
            # 1. Convert to pandas datetime objects uniformly
            df[time_col_name] = pd.to_datetime(df[time_col_name], errors='coerce', utc=True)
            
            # --- Core modification: Handle timezones ---
            # 2. Check if the time column has timezone information (tz-aware)
            if df[time_col_name].dt.tz is not None:
                print(f"Timezone detected in '{os.path.basename(file_path)}', removing...")
                # 3. If it has a timezone, remove it to make it tz-naive
                df[time_col_name] = df[time_col_name].dt.tz_localize(None)
            
            df.dropna(subset=[time_col_name], inplace=True)
            df.set_index(time_col_name, inplace=True)
            
            df_filtered = df.loc[start_date:end_date]
            
            if df_filtered.empty:
                print(f"Warning: No data found between {start_date} and {end_date} in file '{os.path.basename(file_path)}'.")
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
            else:
                x_data = df_filtered.index
                y_data = df_filtered.iloc[:, 0]
                ax.plot(x_data, y_data)
                ax.grid(True)
                
                # --- New/Modified: Enlarge axis tick labels and add labels ---
                # Set X and Y axis labels with specified font size
                ax.set_xlabel('Date', fontsize=LABEL_FONTSIZE)
                ax.set_ylabel('Value', fontsize=LABEL_FONTSIZE)
                
                # Set the font size for axis tick labels
                ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")
            ax.text(0.5, 0.5, 'Processing Error', ha='center', va='center', color='red')
        
        # --- Modification: Enlarge title font size ---
        ax.set_title(f"{sub_labels[i]} {titles[i]}", fontsize=TITLE_FONTSIZE, loc='center')

    fig.autofmt_xdate(rotation=30, ha='right')
    plt.tight_layout(pad=2.0) # Increase padding a bit to prevent label overlap
    
    output_filename = "final_plot_tz_handled_large_font.pdf"
    fig.savefig(output_filename, bbox_inches='tight')
    print(f"Plot successfully saved as: {output_filename}")
    # plt.show()


# --- Main program entry ---
if __name__ == "__main__":
    # Assume your data files are at the correct paths
    # If the files do not exist, ensure the paths are correct or create dummy data first
    my_files = ['data/Canada_photovoltaics_plants/time_series/id_164440.parquet', 
                'data/Germany_Renewable_Power_Grid/impute_data/solar_50Hertz.parquet', 
                'data/NYC_traffic_speed/time_series/id_124.parquet']
    
    # Ensure the directories exist, otherwise creating dummy data will fail
    for file in my_files:
        dir_name = os.path.dirname(file)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Created directory: {dir_name}")

    # To allow the code to run, create dummy files
    try:
        # Try to read; if file doesn't exist, create it
        pd.read_parquet(my_files[0])
    except FileNotFoundError:
        print("Warning: Data files not found, creating dummy data for demonstration...")
        # Create dummy data
        dates_a = pd.to_datetime(pd.date_range(start='2024-11-15', end='2025-01-15', freq='H', tz='UTC'))
        data_a = np.random.rand(len(dates_a)) * 100 + 50 * np.sin(np.linspace(0, 20*np.pi, len(dates_a)))
        df_a = pd.DataFrame({'datetime': dates_a, 'value_a': data_a})
        df_a.to_parquet(my_files[0])

        dates_b = pd.to_datetime(pd.date_range(start='2024-11-15', end='2025-01-15', freq='30T')) # No timezone
        data_b = np.random.rand(len(dates_b)) * 50 + 20
        df_b = pd.DataFrame({'timestamp': dates_b, 'value_b': data_b})
        df_b.to_parquet(my_files[1])

        dates_c = pd.to_datetime(pd.date_range(start='2024-11-15', end='2025-01-15', freq='15T', tz='America/New_York'))
        data_c = np.random.rand(len(dates_c)) * 200 + np.arange(len(dates_c)) * 0.01
        df_c = pd.DataFrame({'time': dates_c, 'value_c': data_c})
        df_c.to_parquet(my_files[2])
        print("Dummy data created.")


    plot_parquets_by_time_range(
        file_paths=my_files,
        start_date='2024-12-01',
        end_date='2025-01-01'
    )