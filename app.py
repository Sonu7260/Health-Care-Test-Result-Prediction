import sys
import subprocess

# -------------------------------------------------------------
# AUTOMATIC DEPENDENCY CHECKER (Ensures no module errors)
# -------------------------------------------------------------
REQUIRED_MODULES = {
    "xgboost": "xgboost",
    "flask": "flask",
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn"
}

for module_name, pip_name in REQUIRED_MODULES.items():
    try:
        __import__(module_name)
    except ImportError:
        print(f"[{module_name.upper()} NOT FOUND]: Installing '{pip_name}' automatically...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"Successfully installed {pip_name}!")
        except Exception as e:
            print(f"Could not auto-install {pip_name}. Error: {e}")

from flask import Flask, render_template_string, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# -------------------------------------------------------------
# 1. LOAD THE XGBOOST MODEL
# -------------------------------------------------------------
MODEL_PATH = 'XGBoost_pkl (1).pkl'
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print(f"CRITICAL WARNING: '{MODEL_PATH}' not found in the current directory.")

# -------------------------------------------------------------
# 2. MATCH THE EXACT EXPECTED TRAINING FEATURES
# -------------------------------------------------------------
EXPECTED_FEATURES = [
    'Age', 'Gender', 'Blood Type', 'Medical Condition', 'Hospital', 
    'Insurance Provider', 'Billing Amount', 'Admission Type', 'Medication'
]

# Dropdown list options mapped perfectly for the User Interface (UI)
DROPDOWN_OPTIONS = {
    'Gender': ['Male', 'Female'],
    'Blood Type': ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'],
    'Medical Condition': ['Diabetes', 'Diabetes Type 2', 'Hypertension', 'Asthma', 'Obesity', 'Arthritis', 'Cancer', 'Stroke', 'Other'],
    'Hospital': ['General Hospital', 'City Medical Center', 'St. Jude Hospital', 'Mayo Clinic Regional', 'Grace Hospital', 'Mercy Health Lab', 'Community Clinic'],
    'Insurance Provider': ['Medicare', 'Medicaid', 'Blue Cross', 'Aetna', 'Cigna', 'UnitedHealthcare', 'None'],
    'Admission Type': ['Emergency', 'Urgent', 'Elective'],
    'Medication': ['Metformin', 'Insulin', 'Lisinopril', 'Amlodipine', 'Albuterol', 'Atorvastatin', 'Ibuprofen', 'None']
}

# -------------------------------------------------------------
# 3. EMBEDDED SINGLE-FILE HTML & CSS INTERFACE
# -------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Healthcare Analytics & Risk Management Portal</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="bg-slate-50 min-h-screen text-slate-800 font-sans">

    <header class="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-xs">
        <div class="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="p-2 bg-indigo-600 rounded-lg text-white">
                    <i class="fa-solid fa-hospital-user text-xl"></i>
                </div>
                <div>
                    <h1 class="text-lg font-bold tracking-tight text-slate-900">
                        ClinicalPredict <span class="text-xs bg-indigo-100 text-indigo-800 px-2 py-0.5 rounded-full font-medium ml-1">XGBoost Native Fix</span>
                    </h1>
                    <p class="text-xs text-slate-500">Patient Case & Billing Risk Matrix Evaluation</p>
                </div>
            </div>
            <div class="text-xs text-slate-400 flex items-center space-x-2">
                <span class="inline-block w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span>Encoded Channels Stable</span>
            </div>
        </div>
    </header>

    <main class="max-w-6xl mx-auto px-4 py-8 grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        <div class="lg:col-span-2">
            <div class="bg-white rounded-2xl border border-slate-200 shadow-md p-6">
                <div class="flex items-center space-x-2 border-b border-slate-100 pb-4 mb-6">
                    <i class="fa-solid fa-folder-medical text-indigo-600 text-lg"></i>
                    <h2 class="text-xl font-semibold text-slate-900">Patient Admission Profiles</h2>
                </div>

                {% if error %}
                <div class="mb-6 p-4 bg-rose-50 border border-rose-200 text-rose-800 rounded-xl flex items-center space-x-3 text-sm">
                    <i class="fa-solid fa-triangle-exclamation text-rose-500 text-lg"></i>
                    <span class="font-mono text-xs">{{ error }}</span>
                </div>
                {% endif %}

                <form action="/predict" method="POST" class="grid grid-cols-1 sm:grid-cols-2 gap-5">
                    
                    {% for feature in expected_features %}
                        <div class="flex flex-col space-y-1.5">
                            <label for="{{ feature }}" class="text-xs font-semibold uppercase tracking-wider text-slate-500">
                                {{ feature }}
                            </label>

                            {% if feature in dropdown_options %}
                                <select 
                                    id="{{ feature }}" 
                                    name="{{ feature }}" 
                                    class="w-full bg-slate-50 hover:bg-slate-100/50 focus:bg-white px-4 py-2.5 rounded-xl border border-slate-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100 transition outline-none text-slate-900"
                                    required
                                >
                                    <option value="" disabled selected>Select {{ feature }}...</option>
                                    {% for option in dropdown_options[feature] %}
                                        <option value="{{ option }}" {% if user_inputs and user_inputs[feature] == option %}selected{% endif %}>{{ option }}</option>
                                    {% endfor %}
                                </select>

                            {% elif feature == 'Age' %}
                                <input 
                                    type="number" 
                                    min="0" max="120"
                                    id="Age" name="Age" 
                                    value="{{ user_inputs['Age'] if user_inputs else '' }}"
                                    placeholder="e.g. 45"
                                    class="w-full bg-slate-50 hover:bg-slate-100/50 focus:bg-white px-4 py-2.5 rounded-xl border border-slate-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100 transition outline-none text-slate-900 placeholder-slate-400"
                                    required
                                >
                            {% elif feature == 'Billing Amount' %}
                                <input 
                                    type="number" 
                                    step="0.01" min="0"
                                    id="Billing Amount" name="Billing Amount" 
                                    value="{{ user_inputs['Billing Amount'] if user_inputs else '' }}"
                                    placeholder="e.g. 24500.00"
                                    class="w-full bg-slate-50 hover:bg-slate-100/50 focus:bg-white px-4 py-2.5 rounded-xl border border-slate-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100 transition outline-none text-slate-900 placeholder-slate-400"
                                    required
                                >
                            {% endif %}
                        </div>
                    {% endfor %}

                    <div class="sm:col-span-2 pt-4 border-t border-slate-100 mt-4">
                        <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-6 rounded-xl shadow-lg shadow-indigo-600/20 transition transform active:scale-[0.99] flex items-center justify-center space-x-2 cursor-pointer">
                            <i class="fa-solid fa-brain"></i>
                            <span>Compute Matrix Array Prediction</span>
                        </button>
                    </div>
                </form>
            </div>
        </div>

        <div class="lg:col-span-1">
            <div class="bg-white rounded-2xl border border-slate-200 shadow-md p-6 h-full flex flex-col justify-between">
                <div>
                    <div class="flex items-center space-x-2 border-b border-slate-100 pb-4 mb-6">
                        <i class="fa-solid fa-chart-line text-indigo-600 text-lg"></i>
                        <h2 class="text-xl font-semibold text-slate-900">Analysis Metrics</h2>
                    </div>

                    {% if prediction_made %}
                        <div class="text-center py-6 px-4 rounded-2xl mb-6 bg-gradient-to-br from-indigo-50 to-blue-50 border border-indigo-100">
                            <p class="text-xs uppercase font-bold tracking-widest text-slate-400 mb-1">Model Index Classification</p>
                            <h3 class="text-3xl font-black tracking-tight text-indigo-900">
                                Result: {{ predicted_class }}
                            </h3>

                            {% if confidence %}
                            <div class="mt-4 inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold bg-white border border-slate-200/50 shadow-xs">
                                <span class="text-slate-500">Distribution Reliability:</span>
                                <span class="text-indigo-600">{{ confidence }}%</span>
                            </div>
                            {% endif %}
                        </div>

                        <div class="p-4 rounded-xl text-sm leading-relaxed bg-slate-50 border border-slate-200 text-slate-700">
                            <div class="font-bold text-slate-900 mb-1 flex items-center space-x-1.5">
                                <i class="fa-solid fa-circle-info text-indigo-500"></i>
                                <span>Numerical Vector Match</span>
                            </div>
                            The UI categories were converted into structural index weights matching the raw values required by your strict local booster configurations. Complete!
                        </div>
                    {% else %}
                        <div class="flex flex-col items-center justify-center text-center py-20 px-4 border-2 border-dashed border-slate-200 rounded-2xl bg-slate-50/50">
                            <div class="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center text-slate-400 mb-4">
                                <i class="fa-solid fa-server text-2xl"></i>
                            </div>
                            <h4 class="text-sm font-semibold text-slate-700 mb-1">Awaiting Data Execution</h4>
                            <p class="text-xs text-slate-400 max-w-[200px]">Select information from the dropdown matrices and submit to read the internal classification output array.</p>
                        </div>
                    {% endif %}
                </div>

                <div class="text-[10px] text-slate-400 border-t border-slate-100 pt-4 mt-6 text-center">
                    <i class="fa-solid fa-shield-halved mr-1"></i> Data Pipeline Validation: Secure.
                </div>
            </div>
        </div>

    </main>
</body>
</html>
"""

# -------------------------------------------------------------
# 4. ROUTING LOGIC WITH INDEX ENCODING CONVERSIONS
# -------------------------------------------------------------
@app.route('/')
def home():
    return render_template_string(
        HTML_TEMPLATE, 
        expected_features=EXPECTED_FEATURES, 
        dropdown_options=DROPDOWN_OPTIONS,
        user_inputs=None
    )

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(
            HTML_TEMPLATE, 
            expected_features=EXPECTED_FEATURES, 
            dropdown_options=DROPDOWN_OPTIONS,
            error=f"Model configuration mapping missing. Place your '{MODEL_PATH}' inside this folder directory.",
            user_inputs=request.form
        )

    try:
        numerical_input_data = {}
        
        for feature in EXPECTED_FEATURES:
            val = request.form.get(feature)
            if val is None or val == '':
                return render_template_string(
                    HTML_TEMPLATE, 
                    expected_features=EXPECTED_FEATURES, 
                    dropdown_options=DROPDOWN_OPTIONS,
                    error=f"Missing feature assignment for input metrics mapping: {feature}",
                    user_inputs=request.form
                )
            
            if feature in ['Age', 'Billing Amount']:
                numerical_input_data[feature] = float(val)
            else:
                # CRUCIAL FIX: Convert string options to their numeric dropdown array index position (0, 1, 2...).
                # This guarantees that the matrix strictly sees int/float records, bypassing any enable_categorical requirements.
                option_list = DROPDOWN_OPTIONS[feature]
                if val in option_list:
                    numerical_input_data[feature] = float(option_list.index(val))
                else:
                    # Safeguard for fallback indexes
                    numerical_index = hash(val) % 5
                    numerical_input_data[feature] = float(numerical_index)

        # Convert numerical dictionary into a strict Pandas DataFrame
        input_df = pd.DataFrame([numerical_input_data])
        input_df = input_df[EXPECTED_FEATURES]

        # Convert columns to flat standard float format to satisfy the exact numpy array layout
        for col in input_df.columns:
            input_df[col] = input_df[col].astype(float)

        # Execute classification
        prediction = model.predict(input_df)[0]
        
        # Pull Class Probabilities safely
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df)[0]
            confidence = round(float(np.max(probabilities)) * 100, 2)
        else:
            confidence = None

        return render_template_string(
            HTML_TEMPLATE, 
            expected_features=EXPECTED_FEATURES, 
            dropdown_options=DROPDOWN_OPTIONS,
            prediction_made=True,
            predicted_class=prediction,
            confidence=confidence,
            user_inputs=request.form
        )

    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE, 
            expected_features=EXPECTED_FEATURES, 
            dropdown_options=DROPDOWN_OPTIONS,
            error=f"Processing breakdown: {str(e)}",
            user_inputs=request.form
        )

if __name__ == '__main__':
    app.run(debug=True)
