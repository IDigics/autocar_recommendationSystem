from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model and files
knn = joblib.load("car_knn_model.joblib")
preprocessor = joblib.load("car_preprocessor.joblib")
car_ids = joblib.load("car_ids.joblib")

# ID ↔ index mappings
index_to_id = {i: car_ids[i] for i in range(len(car_ids))}
id_to_index = {car_ids[i]: i for i in range(len(car_ids))}

# Prepare features
df = pd.read_csv("cleaned_cars_dataset.csv")
df = df.drop(columns=["image url", "offerType"], errors="ignore")
X = preprocessor.transform(df.drop(columns=["id"]))

# FastAPI app
app = FastAPI()

class Request(BaseModel):
    car_id: int
    n: int = 3

@app.post("/recommend")
def recommend(request: Request):
    car_id = request.car_id
    n = request.n

    if car_id not in id_to_index:
        raise HTTPException(
            status_code=404,
            detail=f"Car ID {car_id} not found."
        )

    idx = id_to_index[car_id]
    query = X[idx].reshape(1, -1)
    _, indices = knn.kneighbors(query, n_neighbors=n+1)

    recs = [int(index_to_id[i]) for i in indices[0] if i != idx][:n]
    return recs
