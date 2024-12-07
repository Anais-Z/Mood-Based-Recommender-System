from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api', methods=['GET'])
def api():
    return jsonify({"message": "Welcome to your Flask app!"})

if __name__ == '__main__':
    app.run(debug=True)
