import pickle

with open("test_not_none_data.pkl", 'rb') as f:
    data = pickle.load(f)

print(len(data))
