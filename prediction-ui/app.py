# importing Flask and other modules
import json

import requests
from flask import Flask, request, render_template

# Flask constructor
app = Flask(__name__)


# A decorator used to tell the application
# which URL is associated function
@app.route('/checkbodyfat', methods=["GET", "POST"])
def check_bodyfat():
    if request.method == "POST":
        prediction_input = [
            {
                "Age": int(request.form.get("Age")),  # getting input with name = ntp in HTML form
                "Neck": int(request.form.get("Neck")),  # getting input with name = pgc in HTML form
                "Knee": int(request.form.get("Knee")),
                "Ankle": int(request.form.get("Ankle")),
                "Biceps": int(request.form.get("Biceps")),
                "Forearm": float(request.form.get("Forearm")),
                "Wrist": float(request.form.get("Wrist")),
                "Weight": float(request.form.get("Weight")),
                "Height": float(request.form.get("Height")),
                "Abdomen": float(request.form.get("Abdomen")),
                "Chest": float(request.form.get("Chest")),
                "Hip": float(request.form.get("Hip")),
                "Thigh": int(request.form.get("Thigh"))
            }
        ]
        print(prediction_input)
        # use requests library to execute the prediction service API by sending a HTTP POST request
        # localhost or 127.0.0.1 is used when the applications are on the same machine.
        res = requests.post('http://localhost:5000/bodyfat_predictor/', json=json.loads(json.dumps(prediction_input)))
        print(res.status_code)
        result = res.json()
        return result
    return render_template(
        "user_form.html")  # this method is called of HTTP method is GET, e.g., when browsing the link


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5001)
    app.run(host='0.0.0.0', port=5000)
