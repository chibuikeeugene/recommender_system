from pydantic import BaseModel
from typing import List
import typing as t

class RecommendationResult(BaseModel):
    result: List[str]
    version:t.Any
    error: t.Any