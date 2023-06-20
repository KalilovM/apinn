from fastapi import UploadFile
from pydantic import BaseModel

class FeedbackItem(BaseModel):
		image = UploadFile
		label = int

class PredictionRequest(BaseModel):
		image = str