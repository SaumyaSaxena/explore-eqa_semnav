import pickle
if __name__ == "__main__":

    # Replace 'your_file.pkl' with the path to your .pkl file
    with open('/home/saumyas/Projects/semnav/explore-eqa_semnav/results/vlm_exp_semantic/results.pkl', 'rb') as file:
        data = pickle.load(file)

    # Now 'data' contains the object that was stored in the pickle file
    print(data)
    import ipdb; ipdb.set_trace()