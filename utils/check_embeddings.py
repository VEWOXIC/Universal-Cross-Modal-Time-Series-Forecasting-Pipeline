import joblib

pkl = joblib.load("./data/NYC_traffic_speed/weather/merged_report_embedding/static_info_embeddings.pkl")

print(pkl.keys())

print(pkl["channel_info"])