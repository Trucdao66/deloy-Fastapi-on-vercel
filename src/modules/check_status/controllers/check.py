from fastapi import APIRouter, BackgroundTasks, Depends, Response, status, UploadFile
from typing import Optional
import os,io
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from ..models.models import gauss,check_file
from src.modules.data_processing.data_processing import show_data,FIT,scattering
checkRouter = APIRouter()

@checkRouter.get('/')
def check_status():
    return {"status": "OK"}

@checkRouter.post("/uploadfile")
async def UpLoadFile(file: UploadFile,weight:Optional[int]=5,background:Optional[bool] = False):
    check_file(file.filename, [".txt"])
    contents_bytes = await file.read()
    deatails = show_data(contents_bytes,weight,background)
    return {file.filename : deatails}

@checkRouter.post("/uploadfile_and_Fit")
async def UpLoadFile(file: UploadFile, cantren:int , canduoi:int,weight:Optional[int]=5, background:Optional[bool] = False):
    check_file(file.filename, [".txt"])
    contents_bytes = await file.read()
    deatail = show_data(contents_bytes,weight,background)
    deatails = FIT(deatail,cantren,canduoi)
    return {f"area_fit(low:{canduoi},high:{cantren}),function_fit is gauss. Data, data after fit, mean and sigma of each column in the file ": deatails}

@checkRouter.post("/scattering")
async def UpLoadFile(file: UploadFile,weight:Optional[int]=6, background:Optional[bool] = False):
    check_file(file.filename, [".txt"])
    contents_bytes = await file.read()
    deatail = show_data(contents_bytes,weight,background)
    t = scattering(deatail)
    return t