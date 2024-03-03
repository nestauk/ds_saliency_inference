"""App to visualize saliency maps for images.
To run, use:
streamlit run streamlit_viz.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import hmac
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

st.title("Saliency Map Visualizer")

st.markdown(
    """
    ## MSI-Net Model Documentation
    
    ### Overview of MSI-Net
    The Multi-Scale Integration Network (MSI-Net) is a cutting-edge deep learning model designed for saliency detection in images. Saliency detection aims to identify the most important or attention-grabbing parts of an image, which can be crucial for various applications, including image editing, advertising, and content creation. MSI-Net leverages a sophisticated architecture that processes images at multiple scales, allowing it to capture a broad range of salient features from fine details to significant image components. By integrating information across these scales, MSI-Net achieves high precision in highlighting salient regions of an image, making it a powerful tool for researchers and developers working in the field of computer vision and image processing. Our work on this is based on Alexander Kroner’s work, which can be found in a Github repo [here](https://github.com/alexanderkroner/saliency).

    This is a demo of the Saliency Map Visualizer. To use it, upload an image
    and click the button below. Please note, it may take up to 20 seconds to visualise.
    """
)

# get host from secrets
api_host = st.secrets["api_host"]

uploaded_file = st.file_uploader("Choose an image...", type=(["jpg", "jpeg", "png"]))

if uploaded_file is not None:
    file = {'file': uploaded_file.read()}
    st.write("")
    st.write("Classifying...")
    response = requests.post(api_host, files=file)
    arr = np.asarray(json.loads(response.json()))
    st.write("Done!")
    # Show plt plots
    plt.imshow(Image.open(uploaded_file))
    plt.imshow(arr, alpha=0.6)
    plt.axis('off')
    st.pyplot()
