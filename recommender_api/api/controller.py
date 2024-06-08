from recommender_model_package.recommender_model_package import recommendation as r

from fastapi import APIRouter
from schema.rec_response_schema import RecommendationResult


api_router = APIRouter()

@api_router.post("/recommendations/")
def get_recommendations(user_id :float | int = 15889, service_range: int = 5):
    result = r.get_recommendation(
        uid = user_id, 
        service_range = service_range)
    
    return result
