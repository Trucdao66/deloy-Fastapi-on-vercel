from typing import Union
from src.modules.check_status.controllers.check import checkRouter as check_router

from fastapi import FastAPI

app = FastAPI()

app.include_router(check_router, prefix="/he")

