<h1>Google Research Football</h1>
<br>
Lakshay Dahiya, ldahiya@buffalo.edu
<br>
<h2>Setup</h2>
python3 -m venv venv

source venv/bin/activate

<h2>Installation</h2>
pip -r requirements.txt

<h2>Run</h2>
<h3>Training & Testing</h3>
python main.py
<h3>Testing Trained Models</h3>
python main.py --model < modelName >
<br/>
Model name can be found in /model/configuration.json 


Helper Links:
https://github.com/google-research/football/blob/master/gfootball/doc/scenarios.md
https://github.com/google-research/football/blob/master/gfootball/doc/observation.md
