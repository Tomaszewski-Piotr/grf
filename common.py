import pathlib

csv_suffix = ".csv"
data_path = "data"
verbose = True

out_dir = 'out'
model_suffix = '.joblib'
data_suffix = '_Iq.csv'

# path to current model directory
def out_directory():
    return pathlib.Path(out_dir)

def out_file(filename):
    return pathlib.Path(out_directory(), filename)


def log_verbose(*args):
    if verbose:
        print(*args)

def find_data_csv():
    return pathlib.Path(data_path).glob('x_*' + csv_suffix)