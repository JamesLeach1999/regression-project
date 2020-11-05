import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import ShuffleSplit, train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import pickle
import json

# importing data into pandas frame to make easier to manipulate
df1 = pd.read_csv("bengaluru_house_data.csv")
df1.head()

# group by area type and returns count of each
df1.groupby("area_type")['area_type'].agg("count")

# dropping certain columns from the data frame to make things a bit simpler
df2 = df1.drop(["area_type", "society", "balcony", "availability"], axis="columns")
df2.head()
# head just returs the first few rows

# data cleaning
# this shows how many values in a column are null

# instead of dropping the null vales, you can take a median and fill out the the nulls. since we have 30000 rows we can safely drop a few
df2.isnull().sum()

df3 = df2.dropna()
# returns size column with all of the unique values. since som are 4 bedroom other 4 bhk,we make a new column
df3['size'].unique()
# new column called bhk. get the values from the size column after using a callback to get relevant info out of values
# x is just the value you want to change. this gives two array elements, soyou just want the first. also make it a int
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(" ")[0]))
# pandas basically turns the csv into an object, access the columns like this
df3[df3.bhk > 20]
# one value returned will be 1133 - 1384, we wanna get the average of this
df3.total_sqft.unique()

def is_float(x):
    try:
        # try to convert a value into float, if not a valid value like 1133 - 1384
        float(x)
    except:
        return False
    return True
# ignore non float values. the little sqwiggle reutrns the false values, so invalid ones
# when usng is float as a callback, you dont need to enter the params. this is all data cleaning
# we take the average of values like 1133 - 1384, with others like 4125perch we ignore

# getting the aervage for values like 1133 - 1384
df3[~df3['total_sqft'].apply(is_float)]

def convert_sqft_to_num(x):
    # split to get the 2 different numbers
    tokens = x.split("-")
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2

    try:
        # it will convert whatever you put in into a float
        return float(x)
    except:
        return None
#     this returns 300.00, wont return anthing with strings. we create a new data frame for every step
# convert_sqft_to_num("200 - 400")
# copies, obviouslty
df4 = df3.copy()
# remember these are still the weird values cos of the ~ when we applied it to df3
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
# these are returned as an array so need to access it like one

df5 = df4.copy()
# price per square foot column. this will help us doing outlier cleaning
# making a new column is v easy. we are just doing some adjustements for rupees
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']

# sorting out locations. hw many unique, how many values for each etc. the number of nuique is 1304
# this is an issue cs tax will be different for all these places. a high demimensionality problem
# use an "other" category. find how many data points for a given location

df5.location.unique()
# removes leading or ending space in location
df5.location = df5.location.apply(lambda x: x.strip())

# put values of specific houses under their location. gives a location column
# agg is a function with different param options, count is just one. also sprt them
# lets say any cloation eith less than 10 data points will be put under "other"
location_stats = df5.groupby("location")['location'].agg("count").sort_values(ascending=False)
# len(location_stats[location_stats<=10])

location_less_than_10 = location_stats[location_stats<=10]
# go over all locations, if the name is in the less than 10 bit, turn the location (x) into other. else keep the vaue
# apply is quite literally just call back from js
df5.location = df5.location.apply(lambda x: "other" if x in location_less_than_10 else x)

# dealing with outliers. one good way to do it is with std dev. also domain knowledge, "what is the typical square foot per bedroom"
# if square foot per bedroom is less than 300, somethings wrong so remove the data point
df5[df5.total_sqft/df5.bhk<300].head()
# remember the little sqwiggle is basically like the ! operator, so get all values which are not below 300
df6 = df5[~(df5.total_sqft/df5.bhk<300)]
# check all where the price per square foot is either v high or v low
# describe gives you things like mean, std dev, min, percentiles etc
df6.price_per_sqft.describe()

# making a function to remove extremes based on std dev. will only work on data with a normal dist

def remove_pps_outliers(df):
    # get each row of the csv file. also initialise a new dataframe to add returned rows onto
    dataframe_out = pd.DataFrame()
    for key, subdf in df.groupby("location"):
        # subdf is the sub data frame. for each value (row), check mean and standard
        # if the price is within 1 standard deviation, make a new dataframe using panda and add the row on
        mean = np.mean(subdf.price_per_sqft)
        standard = np.std(subdf.price_per_sqft)
        reduced_dataframe = subdf[(subdf.price_per_sqft>(mean - standard)) & (subdf.price_per_sqft<=(mean + standard))]
        # add row where std dev is alright onto the new dataframe. if not alright just get rid of it
        dataframe_out = pd.concat([dataframe_out, reduced_dataframe], ignore_index=True)
    return dataframe_out


df7 = remove_pps_outliers(df6)

# scatter plot for data visualisation. this entire thing just draws the plot

def plot_scatter_chart(df, location):
    # if the dataframe value matches the location param and is either 2 or 3 bhk, assign it to these vars
    # its weird its kinda like an if statement returning values based on the conditions
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    # configure size of grid, as well as the different points to plot
    matplotlib.rcParams['figure.figsize'] = (15, 10)
    # for each point in the vars (theyre basically arrays) then plot it
    # christ this is alot like pseduo code. get the price value from the bhk row
    plt.scatter(bhk2.total_sqft, bhk2.price, color="blue", label="2 bhk", s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker="+", color="green", label="3 bhk", s=50)
    plt.xlabel("total area")
    plt.ylabel("price per foot")
    plt.title(location)
    plt.legend()
    # need this to actually show the thing
    plt.show()

# this shows a couple weird things, that some 2 beds have a higher value than the 2 beds
# plot_scatter_chart(df7, "Hebbal")

# function for removing the outliers
# oh yeah tey're called dictionaries in python not objects
# we should remove properties wehre for example, the price of a 3 bed is the same as a 2 bed in the same location
# example, remove 2 bhk where the price_per_sqft is less than the mean of a 1 bhk

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    # here location is the 'index' (0=some place, 1=some other place etc), location_df is the row itself
    for location, location_df in df.groupby("location"):
        # for each location dataframe (row), create a new dataframe called bhk
        # initialise emty dictionary
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby("bhk"):
            # again bhk is just the index here, so 0,1,2 etc
            # so put the values in here
            bhk_stats[bhk] = {
                "mean": np.mean(bhk_df.price_per_sqft),
                "std": np.std(bhk_df.price_per_sqft),
                "count": bhk_df.shape[0]
            }
        #     once the first for loop is over, you run it again and exclude location row values where the price
        # per square foot is less than the mean of the bhk value set above
        for bhk, bhk_df in location_df.groupby("bhk"):
            # get all of the properties for the current bhk row
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)

    return df.drop(exclude_indices, axis="index")

df8 = remove_bhk_outliers(df7)
# plot_scatter_chart(df8, "Hebbal")

# dealing with bathrooms. this is for if you have a 2 bed with 4 bathrooms, bit weird so need to remove
# also making histograms here, easier than a scatter. just pur the data array as your first param
# plt.hist(df8.bath, rwidth=0.8)
# plt.xlabel("number o baths")
# plt.ylabel("count")
# plt.show()
# whenever you have more baths than beds, add another 2 beds on. if still more baths then remove
df9 = df8[df8.bath<df8.bhk+2]

# now we can begin the learning, drop unessecary points such as price per sqft or size (4 bedrooms) as these were just for outlier removal
# have to convert text columns to number for tax pruposes, dummy  coding
df10 = df9.drop(["size", "price_per_sqft"], axis="columns")

dummies = pd.get_dummies(df10.location)

# this creates a new dummy column for each location. sets the value to one if the location matches, all others to 0
# concatenate the 2 dfs using columns to do so
# we can live with one less column, so we drop the last one. soall the rest are dummy encoded
df11 = pd.concat([df10, dummies.drop("other", axis="columns")], axis="columns")
# drop the location since its already dummy encoded. its like a pipeline of cleaning, take 12 different steps
df12 = df11.drop("location", axis="columns")
# x should only contain independant vars, price is the dependant
X = df12.drop("price", axis="columns")
y = df12.price
# divide into training and test data set
# 0.2 means you want 20 people of 100 to be test samples, the remaining are training samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

lr_clf = LinearRegression()
# this trains the model, then the score shows how good it is. we get 84% which is pretty good, we change params for the best model
lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test)
# shuffle split will randomise the sample, so each iteration will have different data to train with
# n splits is just how many times to train
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# this is cross validation, not too sure what it means
cross_val_score(LinearRegression(), X, y, cv=cv)

# bunch of different regressions. gridcv runs your model on different regressions and parameters
# finding the best model to run

# specify the model and the params,will check the best params called hyper parameter churnin g
def find_best_model_using_gridsearchcv(X, y):
    algos = {
        "linear_regression": {
            "model": LinearRegression(),
            "params": {
                "normalize": [True, False]
            }
        },
        "lasso": {
            "model": Lasso(),
            "params": {
                "alpha": [1,2],
                "selection": ['random', "cyclic"]
            }
        },
        "decision_tree": {
            "model": DecisionTreeRegressor(),
            "params": {
                "criterion": ['mse', 'friedman_mse'],
                "splitter": ['best', "random"]
            }
        }
    }

    scores = []
    # go through each model and change the input values
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        # config is the indv item in each dictionary algo, name is the "index"
        # whichever has the best score is the one returned
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        # once fit is called, it appends the best scores to the list
        gs.fit(X,y)
        scores.append({
            "model": algo_name,
            "best_score": gs.best_score_,
            "best_params": gs.best_params_
        })
    #     return scores into a dataframe
    return pd.DataFrame(scores, columns=['model', 'best_score', "best_params"])
# the winner is linear regression, with the best params as normalise=false
find_best_model_using_gridsearchcv(X, y)

# takes all these inputs and returns est price
def predict_price(location, sqft, bath, bhk):
    # gives the position of the column, after sqft, bath etc so the first one will start at 5
    loc_index = np.where(X.columns==location)[0][0]

    # for location since there are like 200 columns, when you do X.columns you get all the  columns
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        # set the index to be 1 not 5
        x[loc_index] = 1
    # returning linear regression since its the best model
    return lr_clf.predict([x])[0]
# this returns 83 quid, the predicted price
predict_price("1st Phase JP Nagar", 1000, 3, 3)
# with this the price will be lower as if its a 2 bed the beds will be bigger, if 3 they will be smaller so lower price
# predict_price("1st Phase JP Nagar", 1000, 2, 3)

# exportig to pickle file to be used by the python flask server
with open("model/bangalore_home_prices_model.pickle", "wb") as f:
    # pass the model as an arguement. the size is small as it just stores the params, coefficient, intercept, not the data itself
    # we also need the columns information. to do this export to a json file
    pickle.dump(lr_clf, f)

# turn each column to lower case then export for each column

columns = {
    "data_columns": [col.lower() for col in X.columns]
}
# write to the file and put all the coulmns in
with open("model/columns.json", "w") as f:
    f.write(json.dumps(columns))
# need the json and the pickle to build the flask server

# testing print statement
print(predict_price("1st Phase JP Nagar", 1000, 3, 3))
