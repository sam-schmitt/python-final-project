from flask import Flask, request, render_template
from multiple import test_response

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the data from the form
        data = request.form['data']

        # Use the data in your Python code
        test_response(data)

        return render_template('form.html')
    else:
        # Render the form template
        return render_template('form.html')

