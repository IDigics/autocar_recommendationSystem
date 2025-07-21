from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and data
knn = joblib.load("car_knn_model.joblib")
preprocessor = joblib.load("car_preprocessor.joblib")
car_ids = joblib.load("car_ids.joblib")

# Rebuild ID mappings
index_to_id = {i: car_ids[i] for i in range(len(car_ids))}
id_to_index = {car_ids[i]: i for i in range(len(car_ids))}

# Load the original dataset and re-transform it
import pandas as pd
df = pd.read_csv("cleaned_cars_dataset.csv")
df = df.drop(columns=["image url", "offerType"], errors="ignore")
df_features = df.drop(columns=["id"])
X = preprocessor.transform(df_features)

# FastAPI setup
app = FastAPI(title="Car Recommendation API")

class RecommendationRequest(BaseModel):
    car_id: int
    n: int = 3

@app.post("/recommend")
def recommend_cars(request: RecommendationRequest):
    car_id = request.car_id
    n = request.n

    if car_id not in id_to_index:
        raise HTTPException(status_code=404, detail=f"Car ID {car_id} not found.")
    
    idx = id_to_index[car_id]
    query_vector = X[idx].reshape(1, -1)
    distances, indices = knn.kneighbors(query_vector, n_neighbors=n+1)

    similar_ids = []
    for i in indices[0]:
        if i != idx:
            similar_ids.append(int(index_to_id[i]))
        if len(similar_ids) == n:
            break

    return {"recommended_car_ids": similar_ids}
