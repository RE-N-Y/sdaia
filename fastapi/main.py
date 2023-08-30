import pickle as pkl
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

with open("pipeline.pkl", "rb") as f:
    pipeline = pkl.load(f)

mapping = {
    0: "normal",
    1: "suspect",
    2: "bad"
}


app = FastAPI()

class Input(BaseModel):
    baseline_value:float
    accelerations:float
    fetal_movement:float
    uterine_contractions:float
    light_decelerations:float
    severe_decelerations:float
    prolongued_decelerations:float
    abnormal_short_term_variability:float
    mean_value_of_short_term_variability:float
    percentage_of_time_with_abnormal_long_term_variability:float
    mean_value_of_long_term_variability:float
    histogram_width:float
    histogram_min:float
    histogram_max:float
    histogram_number_of_peaks:float
    histogram_number_of_zeroes:float
    histogram_mode:float
    histogram_mean:float
    histogram_median:float
    histogram_variance:float
    histogram_tendency:float

class Output(BaseModel):
    prediction:str

@app.post("/")
def root(input:Input) -> Output:
    data = pd.DataFrame([input.dict()], columns=pipeline["columns"])
    data = pipeline["scaler"].transform(data)
    [prediction] = pipeline["model"].predict(data)
     
    return { "prediction":mapping[prediction] }    
