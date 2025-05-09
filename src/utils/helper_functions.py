import pickle

def create_directory(dirs):
    import os
    directory_name = ""
    for i,dir in enumerate(dirs):

        directory_name += dirs[i] + "/"
        try:
            os.mkdir(directory_name)
            # print(f"Directory '{directory_name}' created successfully.")
        except:
            pass

def picklit(obj,path,name):
    """
    """

    create_directory([f"../../{path}/"])
    filename = f"../../{path}/{name}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def picklin(path,name):
    with open(f'../../{path}/{name}.pkl', 'rb') as f:
        return pickle.load(f)