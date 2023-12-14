from fastapi.exceptions import HTTPException
import numpy as np
import os,io

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def bac4(x, p0,p1,p2,p3,p4):
    return p0 + p1*x + p2*pow(x,2)+p3*pow(x,3) + p4*pow(x,4)

def two_gauss(x,A1,mu1,sigma1,A2,mu2,sigma2):
    return gauss(x,A1,mu1,sigma1) + gauss(x,A2,mu2,sigma2)

def tong(x,p0,p1,p2,p3,p4,A2,mu2,sigma2,A3,mu3,sigma3,A1,mu1,sigma1):
    return  bac4(x,p0,p1,p2,p3,p4) + two_gauss(x,A2,mu2,sigma2,A3,mu3,sigma3) + gauss(x,A1,mu1,sigma1)

def check_file(file_name: str, allowed_extensions: list):
    ext = os.path.splitext(file_name)[1]
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Only {', '.join(allowed_extensions)} files are allowed")