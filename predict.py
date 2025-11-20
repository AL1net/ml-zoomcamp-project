
import pandas as pd
import pickle

# Load the trained model and vectorizer
input_file = 'Random_Forest_Model.bin'

with open(input_file, 'rb') as f_in:
    dv, rf = pickle.load(f_in)

# client dictionary
client = {'name':'Allison Hill',
          'city':'Mariastad',
          'income':33278,
          'credit_score':584, 
          'loan_amount':15446,
          'years_employed':13,
          'points':45.0,
          'loan_approved': False
         }

# Convert to DataFrame
client_df = pd.DataFrame([client])

# Transform features using the loaded encoder
X_client = dv.transform(client_df.to_dict(orient='records'))

# Make prediction
pred_class = rf.predict(X_client)[0]
pred_prob = rf.predict_proba(X_client)[0, 1]  # probability of positive class

# Display input and results
print("client information:")
print(client_df)

print(f"Predicted class: {pred_class}")
print(f"Predicted probability of loan_approved: {pred_prob:.2f}")

# Define approval if necessary
if pred_class == 1:
    print("Loan_approval should be considered for this client.")
else:
    print("Loan_approval should not be considered for this client.")