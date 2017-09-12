import pickle

with open("not_none_data.pkl", 'rb') as f:
    data = pickle.load(f)

print(len(data))
