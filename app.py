from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Define the absolute path to the pickle file
# pickle_file_path = r'C:\Users\vaibh\OneDrive\Desktop\best_gb_model.pkl'
pickle_file_path = r'best_gb_model.pkl'
# Load the model and column names
model = joblib.load(pickle_file_path)
column_names = ['Material Quantity (gm)', 'Additive Catalyst (gm)', 'Ash Component (gm)', 'Water Mix (ml)', 'Plasticizer (gm)', 'Moderate Aggregator', 'Refined Aggregator', 'Formulation Duration (hrs)']

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle the form submission and display the prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = {}
        for col in column_names:
            value = request.form.get(col)
            if not value:
                return render_template('index.html', prediction_text='Invalid input. Please fill all the fields.')
            
            try:
                data[col] = [float(value)]  # Convert the value to a list with a float
                # Check for NaN and infinite values
                if data[col][0] != data[col][0] or data[col][0] == float('inf'):
                    return render_template('index.html', prediction_text='Invalid input. Please enter a valid number.')
            except ValueError:
                return render_template('index.html', prediction_text='Invalid input. Please enter a valid number.')

        data_df = pd.DataFrame(data, columns=column_names)
        prediction = model.predict(data_df)
        return render_template('index.html', prediction_text=f'Compression Strength Prediction: {prediction[0]} MPa')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

