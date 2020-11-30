from flask import Flask, render_template, request
# from analysis import bostonMap

app = Flask(__name__)

@app.route("/")
def goal():
    return render_template("goal.html") 
  
@app.route('/userinstructions/')
def userinstructions():
    return render_template('userinstructions.html')

@app.route('/results/')
def results():
    return render_template('results.html')   

@app.route('/implementation/')
def implement():
    return render_template('implementation.html')  

@app.route('/attribution/')
def attribution_page():
    return render_template('attribution.html')

@app.route('/evolution/')
def attribution():
    return render_template('evolution.html')

@app.route('/map/')
def bostonMap():
     return render_template("bostonMap.html")