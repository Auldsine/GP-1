from flask import Flask, request, jsonify
from Spam_detection import paragraph_prediction

app = Flask(__name__)

# Define the API endpoint for receiving SMS messages
@app.route('/sms', methods=['POST'])
def detect_spam():
    # Get the SMS message from the request
    message = request.json['message']

    # Make a prediction using the machine learning model
    prediction = paragraph_prediction(message)

    # Return the prediction as a JSON response
    response = {'is_spam': prediction}
    return jsonify(response)

if __name__ == '__main__':
    app.run()
