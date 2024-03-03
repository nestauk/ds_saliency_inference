# Saliency Inference API Template

This is an API and Streamlit app to interact with a saliency model. The API is built using FastAPI and the Streamlit app is built using Streamlit. The API is built to be run in a Docker container.

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8080
```

This will run the FastAPI server on port 8080.

### (Alternative) Run the API in a Docker container


```bash
docker build -t ds-api-template .
docker run -p 8080:8080 ds-api-template
```

You can test this is running by executing the same `curl` command as above, which should return the same response.

NOTE: You will need to have Docker installed on your machine. To install Docker, follow the instructions [here](https://docs.docker.com/get-docker/).

## Run the Streamlit App

Once you've set up the API, you can run the Streamlit app to interact with the API.

To run the Streamlit app, run the following command:

```bash
streamlit run app.py
```

You will need to have Streamlit installed on your machine. To install Streamlit, run the following command:

```bash
pip install streamlit
```

You will also need to update a `secrets.toml` file in a `.streamlit` directory at the root of the repo. This file should contain the following:

```toml
api_host = "http://localhost:8080"
password = "<INSERT DESIRED PASSWORD HERE>"
```
