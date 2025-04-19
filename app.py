from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/number', methods=['POST'])
def handle_number():
    # Get the number from JSON request
    data = request.json
    number = data.get('number')
    
    # Print the number to server console
    print(f"Received number: {number}")
    
    # Return a simple confirmation
    return jsonify({"status": "success", "received": number})

if __name__ == '__main__':
    app.run(debug=True) 