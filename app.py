import gradio as gr
import os
gr.Interface.load("models/segmind/SSD-1B", api_key=os.environ.get("HF_TOKEN")).launch()