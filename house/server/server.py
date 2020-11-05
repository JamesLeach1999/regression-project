# creating flask server backend. its a service which can do http requests
from flask import Flask, request, jsonify
import util as util
# util will handle sending back location data
# print(util.get_estimated_price("1st Phase JP Nagar", 1000, 3, 3))
# have to call load saved artifacts first
util.load_saved_artifacts()
print(util.get_location_names())
app = Flask(__name__)

# make json requests in postman

# exposing http endpoitn, apparently if the function name matches the route name, it just calls the functio
@app.route("/get_location_names", methods=['GET'])

# this is called a routine, get all locations using jsonify
def get_location_names():

    response = jsonify({
        "locations": util.get_location_names()
    })
    # returning response w all locations
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route("/predict_home_price", methods=['GET', 'POST'])

def predict_home_price():
    # implementing a dummy routine. can get the form input from request.form
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        "estimated_price": util.get_estimated_price(location, total_sqft, bhk, bath)
    })

    response.headers.add("Access-Control-Allow-Origin", "*")

    return response

if __name__ == "__main__":
    print("starting flask")
    app.run()