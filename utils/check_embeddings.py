import joblib

pkl = joblib.load("./data/Germany_Renewable_Power_Grid/weather/merged_report_embedding/fast_general_formal_embeddings_2011.pkl")

print(pkl.keys())

print(pkl['201101010000'].shape)