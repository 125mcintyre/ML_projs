def import_dataset(path, name):
    from six.moves import urllib
    from scipy.io import loadmat

    data_raw = loadmat(path)
    data = {
        "data": data_raw["data"].T,
        "target": data_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": name,
        }
    print("Successful Load of: ", data["DESCR"])
    return data
