import joblib

pkl = joblib.load("./data/California_ISO/static_info_embeddings_merged_except_battery.pkl")

print(pkl.keys())

print(pkl["channel_info"]["merged_data"].shape)