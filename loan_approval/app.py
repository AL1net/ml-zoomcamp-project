from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os

app = FastAPI()


@app.on_event("startup")
def load_model():
    model_path = "Random_Forest_Model.bin"
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f_in:
        dv, model = pickle.load(f_in)
    app.state.dv = dv
    app.state.model = model


class LoanApprovalRequest(BaseModel):
    name: str
    city: str
    income: int
    credit_score: int
    loan_amount: int
    years_employed: int
    points: float
    loan_approved: bool = True


@app.post("/predict")
def predict(request: LoanApprovalRequest):
    # Convert request to dictionary format
    client_dict = request.dict()

    # Ensure model is loaded
    try:
        dv = app.state.dv
        model = app.state.model
    except Exception:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        X_client = dv.transform([client_dict])
        pred_class = model.predict(X_client)[0]
        if hasattr(model, "predict_proba"):
            pred_prob = model.predict_proba(X_client)[0, 1]
        else:
            # fallback if model has no predict_proba
            pred_prob = float(pred_class)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "predicted_class": int(pred_class),
        "loan_approved_probability": float(pred_prob),
        "approval_status": "Approved" if int(pred_class) == 1 else "Not Approved"
    }


@app.get("/")
def read_root():
    return {"message": "Loan Approval Prediction API", "version": "1.0"}
