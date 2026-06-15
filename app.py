from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load pre-trained pipeline and XGBoost model at startup
try:
    ohe = joblib.load('models/ohe.pkl')
    scaler = joblib.load('models/scaler.pkl')
    model = joblib.load('models/xgboost_model.pkl')
    expected_features = list(model.feature_names_in_)
    print("Models and pipelines loaded successfully.")
except Exception as e:
    print(f"Error loading models or encoder: {e}")
    ohe, scaler, model, expected_features = None, None, None, []

def predict_churn_by_probability(prob, threshold=0.4):
    """Predicts binary churn class using custom probability threshold."""
    return 1 if prob >= threshold else 0

@app.route('/')
def home():
    return render_template('home2.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation2.html')

@app.route('/predict_model', methods=['GET', 'POST'])
def predict_model():
    if request.method == 'POST':
        try:
            # 1. Extract raw inputs from form
            gender = int(request.form.get('gender', 0))
            senior = int(request.form.get('SeniorCitizen', 0))
            partner_str = request.form.get('Partner', 'No')
            dependents_str = request.form.get('Dependents', 'No')
            
            phone_str = request.form.get('PhoneService', 'No')
            multiple_str = request.form.get('MultipleLines', 'No')
            internet_str = request.form.get('InternetService', 'No')
            
            security_str = request.form.get('OnlineSecurity', 'No')
            backup_str = request.form.get('OnlineBackup', 'No')
            protection_str = request.form.get('DeviceProtection', 'No')
            tech_str = request.form.get('TechSupport', 'No')
            tv_str = request.form.get('StreamingTV', 'No')
            movies_str = request.form.get('StreamingMovies', 'No')
            
            contract = request.form.get('Contract', 'Month-to-month')
            paperless = request.form.get('PaperlessBilling', 'No')
            payment = request.form.get('PaymentMethod', 'Electronic check')
            
            tenure = int(request.form.get('tenure', 0))
            monthly = float(request.form.get('MonthlyCharges', 0.0))
            total = float(request.form.get('TotalCharges', 0.0))

            map_addon = lambda val: 1 if val == 'Yes' else (-1 if val == 'No' else 0)
            
            security = map_addon(security_str)
            backup = map_addon(backup_str)
            protection = map_addon(protection_str)
            tech = map_addon(tech_str)
            tv = map_addon(tv_str)
            movies = map_addon(movies_str)
            
            has_partner = 1 if partner_str == 'Yes' else 0
            has_dependents = 1 if dependents_str == 'Yes' else 0
            has_phoneservice = 1 if phone_str == 'Yes' else 0
            has_multiplelines = 1 if multiple_str == 'Yes' else 0
            
            internet_map = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
            internet_encoded = internet_map.get(internet_str, 0)
            
            is_automatic = 1 if 'automatic' in payment else 0
            
            addons_list = [security_str, backup_str, protection_str, tech_str, tv_str, movies_str]
            product_count = sum(1 for add in addons_list if add == 'Yes')
            
            high_risk = 1 if product_count in [1, 2, 3] else 0
            fully_integrated = 1 if product_count >= 5 else 0
            mod_security = 1 if security_str == 'Yes' else 0


            user_data = {
                'gender': gender,
                'SeniorCitizen': senior,
                'tenure': tenure,
                'OnlineSecurity': security,
                'OnlineBackup': backup,
                'DeviceProtection': protection,
                'TechSupport': tech,
                'StreamingTV': tv,
                'StreamingMovies': movies,
                'Contract': contract,
                'PaperlessBilling': paperless,
                'PaymentMethod': payment,
                'MonthlyCharges': monthly,
                'TotalCharges': total,
                'has_partner': has_partner,
                'has_dependents': has_dependents,
                'has_phoneservice': has_phoneservice,
                'has_multiplelines': has_multiplelines,
                'internet_service_encoded': internet_encoded,
                'is_automatic': is_automatic,
                'Product_Count': product_count,
                'Is_High_Risk_Integration': high_risk,
                'Is_Fully_Integrated': fully_integrated,
                'Mod_Security': mod_security
            }

            df_raw = pd.DataFrame([user_data])


            categorical_cols = ['Contract', 'PaperlessBilling', 'PaymentMethod']
            encoded_data = ohe.transform(df_raw[categorical_cols]).toarray()
            encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(categorical_cols))


            df_processed = pd.concat([df_raw.drop(columns=categorical_cols), encoded_df], axis=1)


            numerical_cols = ['MonthlyCharges', 'TotalCharges']
            df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])

            df_final = df_processed[expected_features]

            churn_prob = float(model.predict_proba(df_final)[0][1])
            prediction = predict_churn_by_probability(churn_prob, threshold=0.4)


            return jsonify({
                'success': True,
                'prediction': prediction,
                'churn_prob': churn_prob
            })

        except Exception as ex:
            print(f"Error during prediction: {ex}")
            return jsonify({
                'success': False,
                'error': str(ex)
            }), 400

    return render_template('predict.html')

if __name__ == "__main__":
    app.run(port=7860, host='0.0.0.0')