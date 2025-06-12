import pickle
import numpy as np
from pathlib import Path
import jax

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

def get_save_path(path,name):
    # Get the path to the current script file
    current_file = Path(__file__).resolve()

    # Get the directory containing the script
    script_dir = current_file.parent.parent.parent

    # Define a relative path to save a file (e.g., "output/results.txt")
    save_path = script_dir / path / name

    # Make sure the directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    return save_path

def picklit(obj,path,name):
    save_path = get_save_path(path,name)
    get_save_path(path,name)
    filename = f"{save_path}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def save_fig(plt,path,name):
    save_path = get_save_path(path,name)
    plt.savefig(save_path)

def picklin(path,name):
    with open(f'../../{path}/{name}.pkl', 'rb') as f:
        return pickle.load(f)

def print_header(headers=None):
    if headers is None:
        l1 = f" Epoch | Training Loss | Validation Accuracy | Runtime "
    else:
        l1 = "|".join(headers)
    l2 =  "-"*len(l1)
    print(f"\n{l1}\n{l2}")


def spaced_string(header,item):
    lh = len(header)
    li = len(str(item))
    s1 = int(np.floor((lh - li)/2))
    s2 = lh-(s1+li)
    string = s1*" "+str(item)+s2*" "
    return string

def print_progress(
        data,
        precision=3,
        headers = [" Epoch "," Training Loss "," Val Accuracy "," Runtime "],
        add_headers = [],
        nextline=True):
    
    data = [f"{dat:.{precision}}" if type(dat)!=int else dat for dat in data]

    for add_h in add_headers:
        headers.append(add_h)

    for extra in range(len(data)-len(headers)):
        headers.append(" Extra ")

    if data[0]==0:
        print_header(headers=headers)

    strings = [spaced_string(headers[i],data[i]) for i,head in enumerate(data)]
    line = "|".join(strings)
    if nextline==True:
        print(line)
    else:
        print(line,end="\r")

def tree_shapes(tree):
    return jax.lax.map(lambda p: p.shape, tree)

def tree_sums(tree):
    return jax.lax.map(lambda p: np.sum(p), tree)

def tree_means(tree):
    return jax.lax.map(lambda p: np.mean(p), tree)

