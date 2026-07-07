from pydantic import BaseModel
from typing import List

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]