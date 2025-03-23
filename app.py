from flask import Flask, request, jsonify
from prophet_models import read_file, train_models, predict_future
import pandas as pd

app = Flask(__name__)

# Load and train once at startup
orders = read_file()
predictions = train_models(orders)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Flask backend is running!"})

@app.route("/predict", methods=["GET"])
def predict():
    # Get parameters
    product_id = request.args.get("product_id")
    date_str = request.args.get("date")

    if not product_id or not date_str:
        return jsonify({"error": "Missing 'product_id' or 'date' in query params"}), 400

    try:
        # Convert date to datetime
        date = pd.to_datetime(date_str)
    except Exception as e:
        return jsonify({"error": f"Invalid date format: {e}"}), 400

    # Get prediction
    result = predict_future(date, product_id, predictions)

    if isinstance(result, dict) and "error" in result:
        return jsonify(result), 404

    # Convert result (DataFrame) to JSON
    return jsonify(result.to_dict(orient="records"))
    

if __name__ == "__main__":
    app.run(debug=True)
