# MBTC 2020 Challenge 2 - Machine Learning Model Deployment

In the MBTC 2020 Challenge 2, we have created and deployed a machine learning model that predicts student profiles based on various academic and personal attributes. This README will guide you through the process of deploying and using the model via the Watson Machine Learning (WML) platform.

## Prerequisites

Before proceeding, ensure you have the following prerequisites:

- Python 3.6 or higher
- Watson Machine Learning service instance credentials
- Watson Machine Learning Python client (`watson-machine-learning-client`)

## Preparation

First, we need to install the required Python libraries and import them. Run the following commands:

```python
!pip install scikit-learn==0.20.0 --upgrade
import json
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from watson_machine_learning_client import WatsonMachineLearningAPIClient
```

## Data Preparation

The dataset for this challenge includes student records with various features. We will use this data to build a machine learning model to predict student profiles.

## Custom Transformations

We've created custom transformations using the `DropColumns` class. This transformation allows us to drop specific columns from the dataset.

To use these custom transformations in the WML pipeline, we need to package them as a Python library. This package can be created using the provided repository: [sklearn_transforms](https://github.com/vnderlev/sklearn_transforms).

### Building the Python Library

1. Fork the [sklearn_transforms](https://github.com/vnderlev/sklearn_transforms) repository.
2. Add your custom transformations to the `sklearn_transformers.py` file.
3. Clone the repository to your local environment.

```python
!git clone https://github.com/vnderlev/sklearn_transforms.git
!cd sklearn_transforms
```

### Zip Your Custom Library

Zip the repository for deployment:

```python
!zip -r sklearn_transforms.zip sklearn_transforms
```

### Install Your Custom Library

Install the custom library using the following command:

```python
!pip install sklearn_transforms.zip
```

## Creating a Pipeline

We will create a pipeline that includes data preprocessing and model training. The pipeline consists of the following stages:

1. Remove specific columns from the dataset.
2. Impute missing values with zeros.
3. Train a Decision Tree Classifier.

```python
# Custom transformation to remove columns
rm_columns = DropColumns(
    columns=["NOME"]
)

# Imputer to fill missing values with zeros
si = SimpleImputer(
    missing_values=np.nan,
    strategy='constant',
    fill_value=0
)

# Define features and target columns
features = [
    "MATRICULA", "NOME", 'REPROVACOES_DE', 'REPROVACOES_EM', "REPROVACOES_MF", "REPROVACOES_GO",
    "NOTA_DE", "NOTA_EM", "NOTA_MF", "NOTA_GO",
    "INGLES", "H_AULA_PRES", "TAREFAS_ONLINE", "FALTAS", 
]
target = ["PERFIL"]

# Prepare data for the pipeline
X = df_data_1[features]
y = df_data_1[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=337)

# Create a pipeline with custom transformations and a Decision Tree Classifier
my_pipeline = Pipeline(
    steps=[
        ('remove_cols', rm_columns),
        ('imputer', si),
        ('dtc', DecisionTreeClassifier()),
    ]
)

# Fit the pipeline (preprocessing and model training)
my_pipeline.fit(X_train, y_train)
```

## Deploying the Model

Now, we will deploy the trained model using Watson Machine Learning.

### Connecting to Watson Machine Learning

To connect to your Watson Machine Learning instance, provide your WML credentials:

```python
wml_credentials = {
    "apikey": "",
    "iam_apikey_description": "",
    "iam_apikey_name": "",
    "iam_role_crn": "",
    "iam_serviceid_crn": "",
    "instance_id": "",
    "url": ""
}

# Initialize the Watson Machine Learning client
clientWML = WatsonMachineLearningAPIClient(wml_credentials)
```

### Create and Store the Custom Library

We'll create a custom library to store our custom transformations:

```python
pkg_meta = {
    clientWML.runtimes.LibraryMetaNames.NAME: "my_custom_sklearn_transform_1",
    clientWML.runtimes.LibraryMetaNames.DESCRIPTION: "A custom sklearn transform",
    clientWML.runtimes.LibraryMetaNames.FILEPATH: "sklearn_transforms.zip",
    clientWML.runtimes.LibraryMetaNames.VERSION: "1.0",
    clientWML.runtimes.LibraryMetaNames.PLATFORM: { "name": "python", "versions": ["3.6"] }
}
custom_package_details = clientWML.runtimes.store_library(pkg_meta)
custom_package_uid = clientWML.runtimes.get_library_uid(custom_package_details)
```

### Create a Custom Runtime

Create a runtime for the custom library:

```python
runtime_meta = {
    clientWML.runtimes.ConfigurationMetaNames.NAME: "my_custom_wml_runtime_1",
    clientWML.runtimes.ConfigurationMetaNames.DESCRIPTION: "A Python runtime with custom sklearn Transforms",
    clientWML.runtimes.ConfigurationMetaNames.PLATFORM: {
        "name": "python",
        "version": "3.6"
    },
    clientWML.runtimes.ConfigurationMetaNames.LIBRARIES_UIDS: [custom_package_uid]
}
runtime_details = clientWML.runtimes.store(runtime_meta)
custom_runtime_uid = clientWML.runtimes.get_uid(runtime_details)
```

### Store the Pipeline in Watson Machine Learning

Store the pipeline definition in Watson Machine Learning:

```python
model_meta = {
    clientWML.repository.ModelMetaNames.NAME: 'desafio-2-mbtc2020-pipeline-1',
    clientWML.repository.ModelMetaNames.DESCRIPTION: "My pipeline for submission",
    clientWML.repository.ModelMetaNames.RUNTIME_UID: custom_runtime_uid
}

stored_model_details = clientWML.repository.store_model(
    model=my_pipeline,
    meta_props=model_meta,
    training_data=None
)
```

### Deploy the Model

Finally, deploy the model for consumption:

```python
model_deployment_details = clientWML.deployments.create(
    artifact_uid=stored_model_details["metadata"]["guid"],
    name="desafio-2-mbtc2020-deployment-1",
    description="Solution for MBTC challenge 2",
    asynchronous=False,
    deployment_type='online',
    deployment_format='Core ML',
    meta_props=model_meta
)
```

## Testing the Deployed Model

You can now test the deployed model by using its endpoint URL:

```python
model_endpoint_url = clientWML.deployments.get_scoring_url(model_deployment_details)
print("API Endpoint URL:
