"""App to visualize saliency maps for images.
To run, use:
streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import io


# Function to load the TensorFlow model
@st.cache_resource
def load_model(model_path):
    return tf.saved_model.load(model_path)

# Function to generate heatmap
def generate_heatmap(model, image_array):
    # Convert image to float32 and normalize if required
    image_array = image_array.astype(np.float32) / 255.0
    input_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    input_tensor = input_tensor[tf.newaxis, ...]
    # Use the model's call function directly
    output = model(input_tensor)["output"]
    type(output)
    heatmap = output.numpy()  # Adjust this if your model output is different
    heatmap = np.squeeze(heatmap)
    return heatmap

# Function to overlay heatmap on the original image
def overlay_heatmap_on_image(image, heatmap):
    # Ensure the heatmap is of type uint8 and single-channel
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 0.6, heatmap_rgb, 0.4, 0)
    return overlay

tab1, tab2 = st.tabs(["App", "About"])

with tab1:
    st.title("Simulated Eye Tracking Model App")

    st.markdown(
    """
    ## Simulated Eye Tracking Demo Model Demo

    This is a demo of the Simulated Eye Tracking Model. To use it, upload an image
    and click the button below. Please note, it may take up to 20 seconds to visualise.

    For more detailed usage documentation, please see [here](https://docs.google.com/document/d/1VWHDdSj6faXrBy1tQd6RVDsDTKYXTQ_nAhqXMAuInw4/edit?usp=sharing).


    """
    )
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Load model
    model_path = "./app"
    model = load_model(model_path)
    model = model.signatures["serving_default"]

    if uploaded_file is not None:
        tensor = tf.io.decode_image(uploaded_file.getvalue(), channels=3)

        inference_shape = (240, 320)
        original_shape = tensor.shape[:2]

        input_tensor = tf.expand_dims(tensor, axis=0)

        input_tensor = tf.image.resize(input_tensor, inference_shape,
                                preserve_aspect_ratio=True)
        saliency = model(input_tensor)["output"]

        saliency = tf.image.resize(saliency, original_shape)

        saliency = tf.squeeze(saliency)
        saliency = saliency.numpy()
        # Overlay heatmap on the image
        image = Image.open(uploaded_file)
        if image.format == 'PNG':
            # Convert PNG to JPEG
            with io.BytesIO() as output:
                image = image.convert('RGB')  # Convert to RGB
                image.save(output, format='JPEG')
                converted_image = output.getvalue()
            converted_image = np.array(Image.open(io.BytesIO(converted_image)))
            overlay_image = overlay_heatmap_on_image(converted_image, saliency)
        else:
            image = np.array(Image.open(uploaded_file))
            overlay_image = overlay_heatmap_on_image(image, saliency)

        # Display the overlay image
        st.image(overlay_image, caption='Overlay Image with Heatmap.', use_column_width=True)



with tab2:
    st.title("About")
    st.markdown(
    """
    ## Simulated Eye Tracking Model App Documentation

    This document has been divided into two sections:
    - A user-guide for those looking to use and understand simulated eye tracking
    - A technical guide for those looking to implement the model

    ### User guide

    #### What is simulated eye tracking?
    Simulated eye tracking (SET) sidesteps the demands of real human eye tracking by simulating what parts of an image have the most salience - that is, which parts of an image attract the most attention. SET outputs an intuitive salience map which is overlaid on the original image, and shows where the most attention grabbing elements are. 

    SET uses a machine learning algorithm that has been trained and validated using an annotated dataset of natural images. It takes an image and analyses both low level visual features (e.g. contrast, colour, shape) and high level visual features (e.g. objects, faces, symbols). 
    How to use simulated eye tracking
    We want you to be creative about how you might apply this tool. Here are some examples to give you a flavour of how you might use it:


    Use it on client resources (e.g. webpages) to add flair or demonstrate the importance of attention when writing a proposal
    Use it on on webpages to highlight how to improve accessibility (i.e. by ensuring that search boxes or CTA engage sufficient attention)
    On photos taken in the field to see if there are distractions, or better placements for items
    To inform the iterative design of visual materials, make sure critical information is prioritised and there isn’t any distracting visual content, and directly compare different designs
    To assess the optimal deployment of visual materials in real-world environments, including reports and data visualisations
    To demonstrate what we mean by ‘Attractive’ when doing education and capability building

    To actually use the tool, simply upload an image, photo, or webpage, and SET will output the image with a salience map overlaid on top.

    Please be aware that the Streamlit app may occasionally "fall asleep" if it has not been used for a while. If you encounter a message indicating that the app is asleep, simply select the option to wake it up. The app should be up and running within a few minutes, ready for you to use.

    #### How to interpret salience maps
    The salience maps generated by SET are relatively intuitive, with red indicating the areas of higher salience, and blue indicating areas of lower salience (with a natural yellow-green gradient in between. However, interpreting these images is a bit of an art, and there are some things to keep in mind:


    Salience maps are normalised to the image - you should not directly compare the colours between two different salience maps generated by SET.
    Having broad coverage of salience across an image is not necessarily a good thing, and could indicate that there are lots of visual elements competing for attention.
    Salience maps don’t contain any information about timing; for example, where someone will look first, or how long they will look at an area.
    High salience does not necessarily reflect positive valence. For example, gruesome or distressing images are likely to attract considerable salience. Likewise, confusing images or text may also attract attention. 

    This means that the interpretation of these maps is not straightforward - you should combine the maps with your understanding of behavioural science, and ideally other data sources, in order to appropriately interpret a salience map.

    #### FAQ
    Why should I use this?
    - A key strategy for redesigning visual information is to make things Attractive (i.e. the ‘A’ in EAST). While we mostly have good intuitions about this, the quality of existing designs and communications shows that many clients do not. If beauty is in the eye of the beholder, it can take clients many rounds of resource-intensive user testing, focus groups, or actual eye tracking to provide more reliable, objective evidence that something is attractive.

    How is the model trained?
    - The real-world datasets SET is trained on are “natural” images in context (see [SALICON](http://salicon.net/) and [MS COCO](https://paperswithcode.com/dataset/coco) datasets). This means things like animals, objects, people, vehicles, landscapes and scenes - essentially the things people look at every day.

    - However, the model is very sophisticated, and mirrors the way the human visual system works. This means that when we apply it to things like letters, it does a very good job of picking up what is visually important - things like images, high contrasts, or meaningful symbols like dollar signs. This also means we can use it in a way that is agnostic to the actual content of the text, and instead focus on how we interpret the visual patterns of attention (for example, left to right reading direction) to ensure priority information aligns with these patterns.

    Where can I find more details of this?
    - Our work on this is based on Alexander Kroner’s work. There is a technical paper that provides details of the model here, and the core model code which can be found in a Github repository [here](https://github.com/alexanderkroner/saliency).

    Are there any examples I can share with clients?
    - Yes! This [deck](https://docs.google.com/presentation/d/1bQDfUXYpwTqe7J23Nz-8CJ4xH3QeZuLvn4p_3f0am-U/edit#slide=id.gf2b770f2bb_1_4) provides an overview of the technique, and provides some examples of how we can use it. You can also point clients toward this [blog](https://www.bi.team/blogs/eye-robot-using-simulated-eye-tracking-as-a-behavioural-research-technique/) post, which explains the technique, and provides some more examples.

    Who should I go to for more information?
    - In the first instance, please contact Bowen (bowen.fung@bi.team).



    ### Technical documentation and guide
    #### Overview
    The Multi-Scale Integration Network (MSI-Net) is a cutting-edge deep learning model designed for saliency detection in images. Saliency detection aims to identify the most important or attention-grabbing parts of an image, which can be crucial for various applications, including image editing, advertising, and content creation. MSI-Net leverages a sophisticated architecture that processes images at multiple scales, allowing it to capture a broad range of salient features from fine details to significant image components. By integrating information across these scales, MSI-Net achieves high precision in highlighting salient regions of an image, making it a powerful tool for researchers and developers working in the field of computer vision and image processing. Our work on this is based on Alexander Kroner’s work, which can be found in a Github repo [here](https://github.com/alexanderkroner/saliency).

    #### Usage

    ##### Accessing the Streamlit App
    Our MSI-Net model is deployed on a user-friendly Streamlit app, available at https://saliencyinference.streamlit.app/. This web application allows users to easily interact with MSI-Net without the need for a deep understanding of its underlying technology. To access the app and utilise the MSI-Net model (specifically the salicon_cpu model), follow the steps below:
    Enter the Password: Upon visiting the link, you will be prompted to enter a password. Please enter the provided password to gain access to the app's functionalities.
    Upload an Image: Once access is granted, you can drag and drop an image into the designated area or use the file explorer to select an image from your device. The app accepts a range of image formats, ensuring flexibility in usage.
    Model Inference: After uploading the image, the MSI-Net model will automatically process the image. This process involves running the model to detect and highlight the most salient regions of the uploaded image, which may take up to 30 seconds.
    View Results: The result will be a heatmap overlaid on the original image, visually representing the areas deemed most salient by the MSI-Net model. This heatmap allows users to easily identify the focal points of the image.
    Note on App Availability
    Please be aware that the Streamlit app may occasionally "fall asleep" if it has not been used for a while. If you encounter a message indicating that the app is asleep, simply select the option to wake it up. The app should be up and running within a few minutes, ready for you to use.

    ##### Running the App Locally
    For users who prefer to run the application locally, the MSI-Net model is designed to work with both a FastAPI backend and a Streamlit frontend. The code for this is hosted [here](https://huggingface.co/JackVines/ds_saliency_inference). Here's how to set it up:

    ##### Requirements
    - Docker (optional for containerized deployment)
    - Python 3.8 or newer
    - FastAPI Backend

    ##### Direct Run:
    1. Ensure Python 3.8 or newer is installed.
    2. Install the required dependencies, defined in the `requirements.txt` file.
    3. Run the FastAPI app using the command `uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8080`.

    ##### Docker Deployment:
    A Dockerfile is included for containerized deployment.
    Build the Docker image with `docker build -t ds-api-template .`
    Run the container using `docker run -p 8080:8080 ds-api-template`

    ##### Streamlit Frontend
    Before running the Streamlit app, ensure a `.streamlit/secrets.toml` file is configured with the necessary password and host (http://localhost:8080) details.
    Launch the Streamlit app with the command `streamlit run app.py`.
    Note: The FastAPI API and Streamlit frontend run separately and must both be active for full functionality. Ensure they are configured to communicate with each other, typically via the host settings in the .streamlit/secrets.toml file.

    ##### Using a Different Model
    The current hosted version of SET uses a specific model (the salicon_cpu model, which is trained on the SALICON dataset, and uses your machine’s CPU). For users interested in deploying alternative models with the MSI-Net application, there is a collection of models accessible through the following Google Drive link: [Model Repository](https://drive.google.com/drive/folders/1GI7i6GpfI-FoklP3vCc6vxe3T9nk3V2n). These models are initially in TensorFlow 1 (TF1) format and must be converted to TensorFlow 2 (TF2) to ensure compatibility with our application framework.
    ###### Model Conversion Steps:
    ###### Access the Model Repository:
    Navigate to the provided [Google Drive link](https://drive.google.com/drive/folders/1GI7i6GpfI-FoklP3vCc6vxe3T9nk3V2n) and download the desired model. Note that each model might have specific characteristics or advantages, so choose the one that best fits your needs. Download your desired model and place it in the repository
    ###### Convert the Model:
    Locate the `convert_model.py` file within the repository. This script is designed to automate the conversion process from TF1 to TF2 format.
    Execute the script with the path to your downloaded model as an input: `python convert_model.py`
    The script will process the input TF1 model and output a saved_model.pb file, which is the converted model in TF2 format.
    ###### Deploy the Converted Model:
    Ensure the `saved_model.pb` file is in the correct location (/app) within your repository, typically having replaced an existing file with the same name.
    ###### Final Steps
    After successfully converting and deploying the new model, restart your application to apply the changes. The MSI-Net application should now be running with the newly integrated model, ready to process images with the updated saliency inference capabilities.
    """
    )
