# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import pickle as pkl
import pandas as pd
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        with open("pipeline.pkl", "rb") as f:
            self.pipeline = pkl.load(f)

        self.mapping = {
            0: "normal",
            1: "suspect",
            2: "bad"
        }

    def predict(
        self,
        baseline_value:float,
        accelerations:float,
        fetal_movement:float,
        uterine_contractions:float,
        light_decelerations:float,
        severe_decelerations:float,
        prolongued_decelerations:float,
        abnormal_short_term_variability:float,
        mean_value_of_short_term_variability:float,
        percentage_of_time_with_abnormal_long_term_variability:float,
        mean_value_of_long_term_variability:float,
        histogram_width:float,
        histogram_min:float,
        histogram_max:float,
        histogram_number_of_peaks:float,
        histogram_number_of_zeroes:float,
        histogram_mode:float,
        histogram_mean:float,
        histogram_median:float,
        histogram_variance:float,
        histogram_tendency:float
    ) -> str:
        """Run a single prediction on the model"""
        values = [baseline_value, accelerations, fetal_movement, uterine_contractions, light_decelerations, severe_decelerations, prolongued_decelerations, abnormal_short_term_variability, mean_value_of_short_term_variability, percentage_of_time_with_abnormal_long_term_variability, mean_value_of_long_term_variability, histogram_width, histogram_min, histogram_max, histogram_number_of_peaks, histogram_number_of_zeroes, histogram_mode, histogram_mean, histogram_median, histogram_variance, histogram_tendency]
        data = pd.DataFrame(
            [{ c:v for c,v in zip(self.pipeline["columns"], values) }],
            columns = self.pipeline["columns"]
        )
        
        data = self.pipeline["scaler"].transform(data)
        [prediction] = self.pipeline["model"].predict(data)
        
        return self.mapping[prediction]