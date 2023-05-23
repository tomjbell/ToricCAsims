import pickle


def save_obj(obj, name, path):
    with open(path + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, path, suffix='.pkl'):
    with open(path + '/' + name + suffix, 'rb') as f:
        return pickle.load(f)