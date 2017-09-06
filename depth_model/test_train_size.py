import pickle

with open("my_all_frames_list.pkl", 'rb') as f:
    data = pickle.load(f)

print(len(data))
