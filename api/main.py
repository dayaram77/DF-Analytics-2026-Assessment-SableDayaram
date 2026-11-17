
from pathlib import Path
from math import ceil
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------
# Load model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "gradient_boosting_pipeline.pkl"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="SwiftShip Customer Intelligence API",
    description="Income-proxy classification and customer prioritization",
    version="2.0.0"
)


# --------------------------------------------------
# Request schemas
# --------------------------------------------------

class CustomerRecord(BaseModel):
    customer_id: Optional[str] = None

    Age: Optional[float] = None
    Workclass: Optional[str] = None
    Education: Optional[str] = None
    Education_num: Optional[float] = None
    Marital_status: Optional[str] = None
    Occupation: Optional[str] = None
    Relationship: Optional[str] = None
    Race: Optional[str] = None
    Sex: Optional[str] = None
    Capital_gain: Optional[float] = None
    Capital_loss: Optional[float] = None
    Hours_per_week: Optional[float] = None
    Native_country: Optional[str] = None


class PrioritizeRequest(BaseModel):
    customers: list[CustomerRecord] = Field(min_length=1)

    # Default selection is top 20%.
    selection_percentage: float = Field(
        default=20.0,
        gt=0,
        le=100
    )


# --------------------------------------------------
# Data preparation
# --------------------------------------------------

def prepare_customer(customer):
    data = customer.model_dump(exclude={"customer_id"})

    # Convert API field names to the original dataset names.
    rename_columns = {
        "Education_num": "Education-num",
        "Marital_status": "Marital status",
        "Capital_gain": "Capital-gain",
        "Capital_loss": "Capital-loss",
        "Hours_per_week": "Hours-per-week",
        "Native_country": "Native-country"
    }

    data = {
        rename_columns.get(key, key): value
        for key, value in data.items()
    }

    # Normalize blank strings and whitespace.
    # Missing values are handled by the pipeline imputers.
    for column, value in data.items():
        if isinstance(value, str):
            value = value.strip()
            data[column] = value if value else None

    return data


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "SwiftShip Customer Intelligence API is running!",
        "model": "Gradient Boosting",
        "status": "Healthy"
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available"
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": "Gradient Boosting"
    }


# --------------------------------------------------
# Single customer prediction
# --------------------------------------------------

@app.post("/predict")
def predict_customer(customer: CustomerRecord):
    try:
        customer_data = pd.DataFrame([
            prepare_customer(customer)
        ])

        probability = float(
            model.predict_proba(customer_data)[0, 1]
        )

        return {
            "customer_id": customer.customer_id,
            "predicted_probability": round(probability, 4),
            "predicted_class": (
                ">50K" if probability >= 0.5 else "<=50K"
            )
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# --------------------------------------------------
# Customer prioritization
# --------------------------------------------------

@app.post("/prioritize")
def prioritize_customers(request: PrioritizeRequest):
    try:
        customers = request.customers

        # Check duplicate customer IDs.
        customer_ids = [
            customer.customer_id.strip()
            for customer in customers
            if customer.customer_id
            and customer.customer_id.strip()
        ]

        if len(customer_ids) != len(set(customer_ids)):
            raise HTTPException(
                status_code=400,
                detail="Duplicate customer IDs are not allowed."
            )

        # Prepare input data.
        customer_data = pd.DataFrame([
            prepare_customer(customer)
            for customer in customers
        ])

        # Predict probabilities.
        probabilities = model.predict_proba(customer_data)[:, 1]

        results = pd.DataFrame({
            "customer_id": [
                customer.customer_id for customer in customers
            ],
            "predicted_probability": probabilities
        })

        # Rank all customers.
        results = results.sort_values(
            by="predicted_probability",
            ascending=False,
            kind="stable"
        ).reset_index(drop=True)

        results["priority_rank"] = range(1, len(results) + 1)

        # Calculate configurable selection size.
        selection_count = ceil(
            len(results) * request.selection_percentage / 100
        )

        top_customers = results.head(selection_count)

        return {
            "total_customers": len(results),
            "selection_percentage": request.selection_percentage,
            "selected_customer_count": selection_count,

            # Selected top percentage.
            "prioritized_customers": (
                top_customers.to_dict(orient="records")
            ),

            # Complete ranked customer list.
            "all_ranked_customers": (
                results.to_dict(orient="records")
            )
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
