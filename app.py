from flask import Flask, render_template, request
# from analysis import bostonMap

app = Flask(__name__)

@app.route("/")
def goal():
    return render_template("goal.html") 
  
@app.route('/userinstructions.html/')
def userinstructions():
    return render_template('userinstructions.html')

@app.route('/results.html/')
def results():
    return render_template('results.html')   

@app.route('/implementation.html/')
def implement():
    return render_template('implementation.html')  

@app.route('/attribution.html/')
def attribution_page():
    return render_template('attribution.html')

@app.route('/evolution.html/')
def attribution():
    return render_template('evolution.html')

@app.route('/map.html/')
def bostonMap():
     return render_template("bostonMap.html")