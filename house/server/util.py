# global vars to store the values to be used in server
import json
import pickle
import numpy as np


__locations = None
__data_columns = None
__model = None

# make routine to get price given the params

def get_estimated_price(location, sqft, bhk, bath):
    # we are now dealing with a list so cant use np.where
    # loc_index = np.where(X.columns == location)[0][0]
    # use index to find location of specific value, much easier than js. if not found throw error
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    # here the number of 0s will be equal to length of data columns
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        # if the location matches, set the value to one, if not then set all others to 0
        x[loc_index] = 1

    # predict works on sklearn models. takes an input x and returns output estimated price, takes a 2d array
    # this will be a float so round to 2 decimal places
    return round(__model.predict([x])[0], 2)



def load_saved_artifacts():
    print("loading saved artifacts")
    global __data_columns
    global __locations
    global __model

    with open("./artifacts/columns.json", "r") as f:
        # whatever object is loaded here will be converted into a dictionary
        __data_columns = json.load(f)['data_columns']
#         use python index slicing to get everything after the bhk
        __locations = __data_columns[3:]
    # load pickle into model. rb is just read binary
    with open("./artifacts/bangalore_home_prices_model.pickle", "rb") as f:
        __model = pickle.load(f)
    print("loading artifacts done")

# dosent matter what order you make the functions in, just the order you call them

def get_location_names():


    # should read columns.json and return all the locations, starting after the bhk column
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == "__main__":
    load_saved_artifacts()

    print(get_location_names())
    print(get_estimated_price("1st Phase JP Nagar", 1000, 3, 3))
    print(get_estimated_price("1st Phase JP Nagar", 1000, 2, 2))
    print(get_estimated_price("Kalhalli", 1000, 3, 2)) #other location