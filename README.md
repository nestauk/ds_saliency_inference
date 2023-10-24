# DS API Template

This is a template repo structure for developing a data science API. It is based on the [FastAPI](https://fastapi.tiangolo.com/) framework.

The setup is aimed towards inference of a machine learning model, but can be easily adapted to other use cases, such as data curation.

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8080
```

This will run the FastAPI server on port 8080. You can test the default app is running by running this code from the command line:

```bash
curl \
  --header "Content-Type: application/json" \
  --request POST \
  --data '{"data":"MSFT"}' \
  http://localhost:8080/predict
```

This should return `{"prediction":0}%`.

### Run the API in a Docker container

```bash
docker build -t ds-api-template .
docker run -p 8080:8080 ds-api-template
```

You can test this is running by executing the same `curl` command as above, which should return the same response.

NOTE: You will need to have Docker installed on your machine. To install Docker, follow the instructions [here](https://docs.docker.com/get-docker/).

## Development

You will need to update the code in the model.py file to include your own model. The default model simply takes a string input, and returns the value of 0. Depending on the complexity of your model, you may need to add subdirectories and files to the `app` directory.

Ideally, you will want to update the `infer` function in the `model.py` file to include your own model. The `infer` function should take necessary parameters as input, and return a `prediction` as output. Remember to update the `infer` function docstring to explain the input and output formats, and the `requirements.txt` file to include any additional dependencies.

You will also need to update the Data class and predict function in the `main.py` file to include your own data. The Data class should contain the data required to make a prediction.

NOTE: This is not the place to train your model. You should train your model in a separate repo, and then save the model files to the `app` directory. You can then load the model files in the `infer` function. This is to keep this repo as lightweight as possible, as the nature of a repo optimised for inference is very different to one optimised for training.

## Deployment

Depending on your use case, your deployment may look different.
By default, all repos should be pushed to our Github, and also our HuggingFace organisational hub.

From there, we may take a number of routes depending on the use case:

### Open Source

If all we aim to do is Open Source the model, then we can simply push the model to its own repo on the HuggingFace Hub.

This README should be updated to explain the model, how to use it and an evaluation of its performance, such as via a model card. It may also include details relating to any model files not in the repo that need to be added in order to use the model, and how to access these.

Depending on your audience, they may be able to make more use of the `infer` function via Python, and so this should be documented here too. Including the Dockerfile allows users to run the API locally regardless of their setup/chosen language.

### Inference (Open or Private)

If we aim to use the model for inference, then we will need to deploy the API. This can be done in a number of ways, but the easiest is to use the Dockerfile to build a Docker image, and then deploy this image to an EC2 instance, using API Gateway to route requests to the EC2 instance. Consult the DE team for further internal guidance on this process.
