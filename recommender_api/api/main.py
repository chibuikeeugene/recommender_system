
# import os,sys

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from recommender_model_package.recommender_model_package import recommendation as r



app = FastAPI(
    title="Recommendation Service",
    description="Recommendation Service",
    version="0.1.0"
)


@app.get("/")
def read_root():
    return "Welcome to Santander Recommendation Service"


@app.get("/recommendations/")
def get_recommendations(user_id: float, service_range: int):
    result = r.get_recommendation(
        uid = user_id, 
        service_range = service_range)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)