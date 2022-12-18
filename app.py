from flask import Flask, request, render_template
from multiple import ai_test_response
from single import normal_test_response

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the data from the form
        data = request.form['data']

        
        usingAI = request.form.getlist('aicheck')
        if usingAI:
            ai_test_response(data)
        else:
            print("here")
            normal_test_response(data)
        
        

        return render_template('form.html')
    else:
        # Render the form template
        return render_template('form.html')

