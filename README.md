# Simulated Eye Tracking Inference App

This is a Streamlit app to interact with a saliency eye tracking simulation model. We have a live version available, but if you want to run a local version of the app, follow the steps below. You will need to be using Python3.10 environment for the app to work.

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit App

To run the Streamlit app locally, run the following command:

```bash
streamlit run app.py
```
You will need to have an appropriate `.pb` model file in your `app` directory for the app to work. Reach out to jack.vines@nesta.org.uk or bowen.fung@bi.team to get access to this if required.

The app should open automatically in your browser when you run the above, but can also be accessed from `http://localhost:8501/`.

## Generating a Model File

If you want to use a different model, you can do so following these steps.

1. Find your desired model in [this](https://drive.google.com/drive/folders/1GI7i6GpfI-FoklP3vCc6vxe3T9nk3V2n) folder (the default we use is the `model_salicon_cpu` model.)
2. Put the model file in the app directory and update the `graph_pb` variable in the `convert_model.py` file to point to the model file.
3. Run `python convert_model.py`. This will produce a new model file in the `app` directory called `saved_model.pb`.
4. Run the Streamlit app, which will now use your new model.

The purpose of the above steps is to convert the models in the Google drive folder from Tensorflow v1 format to Tensorflow v2 format. This is important as trying to use v1 formats causes a lot of Python dependancy issues with other packages.