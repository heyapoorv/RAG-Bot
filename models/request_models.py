from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5