# Mindgard (Take Home Test)

## Creating a new model

Create a folder in the directory [`models/`](/models/)

```bash
model_name=example_model
mkdir models/${model_name}
```

---

Implement your model in your language of choice

```bash
cd models/${model_name}
mkdir source
nano source/main.py
# ...
```

---

Create a model training pipeline configuration file (see [this example](/models/mnist-cnn/azure/pipeline.yaml)).

Additional documentation available [here](https://learn.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines?view=azureml-api-2).

**Note:** it is important that the pipeline yaml exist in a folder named `azure`!

```bash
mkdir azure
nano azure/pipeline.yaml
```

---

Create a scoring script (see [this example](/models/mnist-cnn/source/score.py)).

Additional documentation available [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&tabs=azure-cli)

```bash
nano source/score.py
```

---

Create a model deployment endpoint file (see [this example](/models/mnist-cnn/azure/deployment.yaml))

```bash
nano source/deployment.yaml
```
