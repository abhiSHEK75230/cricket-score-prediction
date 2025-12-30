from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)
MODEL_PATH = 'pipe.pkl'

class DummyPipe:
    def __init__(self):
        self.team_strength = {
            'Australia': 2, 'India': 3, 'Bangladesh': -1, 'New Zealand': 1,
            'South Africa': 2, 'England': 2, 'West Indies': 0, 'Afghanistan': -2,
            'Pakistan': 1, 'Sri Lanka': -1
        }

    def predict(self, X):
        preds = []
        for _, row in X.iterrows():
            current = float(row.get('current_score', 0))
            balls_left = float(row.get('balls_left', 0))
            wickets_left = float(row.get('wickets_left', row.get('wicket_left', 10)))
            crr = float(row.get('crr', row.get('current_run_rate', max(1.0, current/1.0))))
            last5 = float(row.get('last_five', 0))

            if wickets_left <= 0:
                expected_scoring_rate = 0.8 * crr
            else:
                expected_scoring_rate = crr * (0.9 + 0.01 * min(wickets_left, 10))
            overs_remaining = max(0.0, balls_left / 6.0)
            expected_additional = expected_scoring_rate * overs_remaining

            bat = row.get('batting_team', '')
            bowl = row.get('bowling_team', '')
            bat_str = self.team_strength.get(bat, 0)
            bowl_str = self.team_strength.get(bowl, 0)
            team_influence = 0.6 * bat_str - 0.4 * bowl_str

            city_bias = (hash(str(row.get('city', ''))) % 7 - 3) * 0.3

            raw_pred = current + expected_additional + 0.6 * last5 + team_influence + city_bias
            final_pred = max(current, raw_pred)
            preds.append(int(round(final_pred)))
        return preds

pipe = None
try:
    with open(MODEL_PATH, 'rb') as f:
        pipe = pickle.load(f)
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: could not load '{MODEL_PATH}' ({type(e).__name__}: {e}). Using DummyPipe fallback.")
    pipe = DummyPipe()
    try:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(pipe, f)
        print(f"Saved DummyPipe to {MODEL_PATH}")
    except Exception as se:
        print(f"Could not save DummyPipe to {MODEL_PATH}: {se}")

teams = [
    'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa',
    'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka'
]

cities = [
    'Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town',
    'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban',
    'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion',
    'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton',
    'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi',
    'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff',
    'Christchurch', 'Trinidad'
]

def parse_overs(overs_value):
    """
    Parse overs input (accepts '10', '10.4' meaning 10 overs + 4 balls,
    or decimal like 10.40). Returns (balls_bowled:int, overs_for_crr:float).
    """
    if overs_value is None or str(overs_value).strip() == '':
        return 0, 0.0
    s = str(overs_value).strip()
    try:
        if '.' in s:
            left, right = s.split('.')
            over_part = int(left)
            ball_part = int(right)
            if 0 <= ball_part <= 5:
                balls_bowled = over_part * 6 + ball_part
            else:
                frac = float('0.' + right)
                balls_from_frac = int(round(frac * 6))
                balls_bowled = over_part * 6 + balls_from_frac
        else:
            over_part = int(s)
            balls_bowled = over_part * 6
    except Exception:
        try:
            overs_float = float(s)
            whole = int(overs_float)
            frac = overs_float - whole
            balls_from_frac = int(round(frac * 6))
            balls_bowled = whole * 6 + balls_from_frac
        except Exception:
            balls_bowled = 0
    overs_for_crr = balls_bowled / 6.0 if balls_bowled > 0 else 0.0
    return int(balls_bowled), float(overs_for_crr)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            batting_team = request.form['batting_team']
            bowling_team = request.form['bowling_team']
            city = request.form['city']
            current_score = int(request.form['current_score'])
            overs_input = request.form['overs']
            wickets = int(request.form['wickets'])
            last_five = int(request.form['last_five'])

            
            balls_bowled, overs_for_crr = parse_overs(overs_input)
            balls_left = max(0, 120 - balls_bowled)

        
            wickets_left = max(0, 10 - wickets)
            wicket_left = wickets_left

            crr = (current_score / overs_for_crr) if overs_for_crr > 0 else 0.0
            current_run_rate = crr

            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [city],
                'current_score': [int(current_score)],
                'balls_left': [int(balls_left)],
                'wickets_left': [int(wickets_left)],
                'wicket_left': [int(wicket_left)],
                'crr': [float(crr)],
                'current_run_rate': [float(current_run_rate)],
                'last_five': [int(last_five)]
            })

            for col in ['current_score','balls_left','wickets_left','wicket_left','last_five']:
                input_df[col] = input_df[col].astype(int)
            for col in ['crr','current_run_rate']:
                input_df[col] = input_df[col].astype(float)

            result = pipe.predict(input_df)
            prediction = int(result[0])

        except Exception as e:
            error = str(e)

    return render_template('index.html', teams=sorted(teams), cities=sorted(cities), prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
