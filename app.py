from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Flask backend is running!"})

@app.route("/predict", methods=["GET"])
def predict():
    # Extract 'product_id' from query parameters
    product_id = request.args.get("product_id")

    if product_id:
        # Replace this with your actual model inference later
        result = {
            "product_id": product_id,
            "predicted_quantity": 123.45  # dummy prediction
        }
        return jsonify(result)
    else:
        return jsonify({"error": "Missing 'product_id' query parameter"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)