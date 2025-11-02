import gradio as gr
from joblib import load
import pandas as pd

model = load("./models/car_value_model.joblib")
encoders = load("./encoders/encoders.joblib")

category = ["brand", "fuel_type", "transmission", "ext_col", "int_col", "accident", "clean_title"]


def predict_car_value(brand, model_year, mileage, fuel_type, transmission, ext_col, int_col, accident, clean_title, horse_powers, litres, cylinders):
    try:
        data = pd.DataFrame([{
            "brand": brand,
            "model_year": model_year,
            "milage": mileage,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "ext_col": ext_col,
            "int_col": int_col,
            "accident": accident,
            "clean_title": clean_title,
            "horse_powers": horse_powers,
            "litres": litres,
            "cylinders": cylinders
        }])

        for col, encoder in zip(category, encoders):
            data[col] = encoder.transform(data[col])

        predicted_price = model.predict(data)[0]

        return f"${predicted_price:.2f}"
    except Exception as e:
        return str(e)

interface = gr.Interface(
    fn=predict_car_value,
    inputs=[
        gr.Textbox(label="Brand"),
        gr.Number(label="Model Year"),
        gr.Number(label="Mileage"),
        gr.Textbox(label="Fuel Type"),
        gr.Textbox(label="Transmission"),
        gr.Textbox(label="Exterior Color"),
        gr.Textbox(label="Interior Color"),
        gr.Radio(["None reported", "At least 1 accident or damage reported"], label="Accident"),
        gr.Radio(["Yes", "No"], label="Clean Title"),
        gr.Number(label="Horse Power"),
        gr.Number(label="Litres"),
        gr.Number(label="Cylinders"),
    ],
    outputs=gr.Textbox(label="Estimated Price"),
    title="Car Value Estimator",
    description="Provide car details to get an estimated value."
)

if __name__ == "__main__":
    interface.launch()
    