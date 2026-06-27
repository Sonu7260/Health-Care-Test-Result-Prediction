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
# 2. DEFINE INPUT FEATURES MAPPING
# -------------------------------------------------------------
# Adjust the keys ('age', 'bmi', etc.) to match the exact column names 
# your model was trained on.
FEATURE_MAPPING = {
    'age': 'Age (Years)',
    'bmi': 'Body Mass Index (BMI)',
    'blood_pressure': 'Systolic Blood Pressure (mmHg)',
    'cholesterol': 'Total Cholesterol (mg/dL)',
    'glucose': 'Fasting Blood Glucose (mg/dL)',
    'heart_rate': 'Resting Heart Rate (bpm)'
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
    <title>Healthcare Diagnostics Portal</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="bg-slate-50 min-h-screen text-slate-800 font-sans">

    <header class="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-xs">
        <div class="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="p-2 bg-teal-500 rounded-lg text-white">
                    <i class="fa-solid fa-heart-pulse text-xl animate-pulse"></i>
                </div>
                <div>
                    <h1 class="text-lg font-bold tracking-tight text-slate-900">
                        HealthPredict <span class="text-xs bg-teal-100 text-teal-800 px-2 py-0.5 rounded-full font-medium ml-1">AI Engine</span>
                    </h1>
                    <p class="text-xs text-slate-500">XGBoost Diagnostics Analysis</p>
                </div>
            </div>
            <div class="text-xs text-slate-400 flex items-center space-x-2">
                <span class="inline-block w-2 h-2 rounded-full bg-emerald-500"></span>
                <span>Model Connected</span>
            </div>
        </div>
    </header>

    <main class="max-w-6xl mx-auto px-4 py-8 grid grid-cols-1 md:grid-cols-3 gap-8">
        
        <div class="md:col-span-2">
            <div class="bg-white rounded-2xl border border-slate-200 shadow-md p-6">
                <div class="flex items-center space-x-2 border-b border-slate-100 pb-4 mb-6">
                    <i class="fa-solid fa-clipboard-user text-teal-600 text-lg"></i>
                    <h2 class="text-xl font-semibold text-slate-900">Patient Test Metrics</h2>
                </div>

                {% if error %}
                <div class="mb-6 p-4 bg-rose-50 border border-rose-200 text-rose-800 rounded-xl flex items-center space-x-3 text-sm">
                    <i class="fa-solid fa-triangle-exclamation text-rose-500 text-lg"></i>
                    <span>{{ error }}</span>
                </div>
                {% endif %}

                <form action="/predict" method="POST" class="grid grid-cols-1 sm:grid-cols-2 gap-5">
                    {% for key, label in features.items() %}
                    <div class="flex flex-col space-y-1.5">
                        <label for="{{ key }}" class="text-xs font-semibold uppercase tracking-wider text-slate-500">
                            {{ label }}
                        </label>
                        <input 
                            type="number" 
                            step="any" 
                            id="{{ key }}" 
                            name="{{ key }}" 
                            value="{{ user_inputs[key] if user_inputs else '' }}"
                            placeholder="Enter lab result..."
                            class="w-full bg-slate-50 hover:bg-slate-100/50 focus:bg-white px-4 py-2.5 rounded-xl border border-slate-200 focus:border-teal-500 focus:ring-2 focus:ring-teal-100 transition outline-none text-slate-900 placeholder-slate-400"
                            required
                        >
                    </div>
                    {% endfor %}

                    <div class="sm:col-span-2 pt-4 border-t border-slate-100 mt-4">
                        <button type="submit" class="w-full bg-teal-600 hover:bg-teal-700 text-white font-semibold py-3 px-6 rounded-xl shadow-lg shadow-teal-600/20 transition transform active:scale-[0.99] flex items-center justify-center space-x-2 cursor-pointer">
                            <i class="fa-solid fa-wand-magic-sparkles"></i>
                            <span>Run AI Risk Evaluation</span>
                        </button>
                    </div>
                </form>
            </div>
        </div>

        <div class="md:col-span-1">
            <div class="bg-white rounded-2xl border border-slate-200 shadow-md p-6 h-full flex flex-col justify-between">
                <div>
                    <div class="flex items-center space-x-2 border-b border-slate-100 pb-4 mb-6">
                        <i class="fa-solid fa-chart-pie text-teal-600 text-lg"></i>
                        <h2 class="text-xl font-semibold text-slate-900">Analysis Output</h2>
                    </div>

                    {% if prediction_made %}
                        <div class="text-center py-6 px-4 rounded-2xl mb-6 bg-gradient-to-br 
                            {% if result_meta.color == 'success' %} from-emerald-50 to-teal-50 border border-emerald-200/60 {% else %} from-rose-50 to-orange-50 border border-rose-200/60 {% endif %}">
                            
                            <p class="text-xs uppercase font-bold tracking-widest text-slate-400 mb-1">Predicted Assessment</p>
                            <h3 class="text-2xl font-black tracking-tight 
                                {% if result_meta.color == 'success' %} text-emerald-700 {% else %} text-rose-700 {% endif %}">
                                {{ result_meta.status }}
                            </h3>

                            {% if confidence %}
                            <div class="mt-4 inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold bg-white/80 border border-slate-200/50 shadow-xs">
                                <span class="text-slate-500">Confidence Score:</span>
                                <span class="{% if result_meta.color == 'success' %} text-emerald-600 {% else %} text-rose-600 {% endif %}">{{ confidence }}%</span>
                            </div>
                            {% endif %}
                        </div>

                        <div class="p-4 rounded-xl text-sm leading-relaxed 
                            {% if result_meta.color == 'success' %} bg-emerald-50/50 border border-emerald-100 text-emerald-800 {% else %} bg-rose-50/50 border border-rose-100 text-rose-800 {% endif %}">
                            <div class="font-bold mb-1 flex items-center space-x-1.5">
                                <i class="fa-solid {% if result_meta.color == 'success' %} fa-circle-check {% else %} fa-circle-exclamation {% endif %}"></i>
                                <span>Clinical Action Mapping</span>
                            </div>
                            {{ result_meta.alert }}
                        </div>
                    {% else %}
                        <div class="flex flex-col items-center justify-center text-center py-16 px-4 border-2 border-dashed border-slate-200 rounded-2xl bg-slate-50/50">
                            <div class="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center text-slate-400 mb-4">
                                <i class="fa-solid fa-stethoscope text-2xl"></i>
                            </div>
                            <h4 class="text-sm font-semibold text-slate-700 mb-1">Awaiting Patient Data</h4>
                            <p class="text-xs text-slate-400 max-w-[200px]">Fill out the patient diagnostic health records to view results here.</p>
                        </div>
                    {% endif %}
                </div>

                <div class="text-[10px] text-slate-400 border-t border-slate-100 pt-4 mt-6 text-center">
                    <i class="fa-solid fa-shield-halved mr-1"></i> Data evaluated in secured runtime.
                </div>
            </div>
        </div>

    </main>
</body>
</html>
"""

# -------------------------------------------------------------
# 4. ROUTING LOGIC
# -------------------------------------------------------------
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, features=FEATURE_MAPPING)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, features=FEATURE_MAPPING, error="Model file is missing on this server.")

    try:
        input_data = {}
        # Parse inputs based on feature keys
        for feature_key in FEATURE_MAPPING.keys():
            val = request.form.get(feature_key)
            if val is None or val == '':
                return render_template_string(HTML_TEMPLATE, features=FEATURE_MAPPING, 
                                              error=f"Missing input field: {FEATURE_MAPPING[feature_key]}")
            input_data[feature_key] = float(val)

        # Structure input matching column configurations for XGBoost
        input_df = pd.DataFrame([input_data])
        
        # Execute Prediction
        prediction = model.predict(input_df)[0]
        
        # Pull Class Probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df)[0]
            confidence = round(float(np.max(probabilities)) * 100, 2)
        else:
            confidence = None

        # Output UI Visual Configurations Mapping
        result_mapping = {
            0: {
                "status": "Normal / Low Risk", 
                "color": "success", 
                "alert": "Biomarkers are tracking stable. Routine preventative screening recommended."
            },
            1: {
                "status": "Elevated / Warning Status", 
                "color": "danger", 
                "alert": "Abnormal thresholds identified. Diagnostic results flag a high correlation risk. Clinical follow-up advised."
            }
        }
        
        # Fallback dictionary mapping for index classification handling safely
        result_meta = result_mapping.get(int(prediction), {
            "status": f"Class {prediction} Detected", 
            "color": "danger", 
            "alert": "Evaluation completed successfully."
        })

        return render_template_string(
            HTML_TEMPLATE, 
            features=FEATURE_MAPPING, 
            prediction_made=True,
            result_meta=result_meta,
            confidence=confidence,
            user_inputs=request.form
        )

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, features=FEATURE_MAPPING, error=f"Processing breakdown: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
