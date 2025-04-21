import gradio as gr
import random
import pandas as pd
import numpy as np

# Rule-based function for demo
def predict(transaction_id, amount, credit_score):
    fraud_conditions = [amount > 30000, credit_score < 500]
    if sum(fraud_conditions) >= 1:
        return "Fraud", round(random.uniform(0.8, 1.0), 2)
    return "Not Fraud", round(random.uniform(0.0, 0.5), 2)

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Transaction ID"),
        gr.Number(label="Amount"),
        gr.Number(label="Credit Score")
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Probability")
    ],
    title="Transaction Fraud Detection",
    description="Enter transaction details to predict if it is fraudulent."
)

iface.launch()
