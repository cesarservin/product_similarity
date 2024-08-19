# product_similarity

## Description
What `problem` does it `solve`?

Finds `similar products` based on their `text` based descriptions to find `competition, complements, and niches` in the market


![Logo](docs/img/product_similarity_logo.jpg)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

[![View Project](https://img.shields.io/badge/Material-View_Project-purple?logo=MaterialforMKDOCS)](https://cesarservin.com/product_similarity/index.html)
[![Open Notebook](https://img.shields.io/badge/Jupyter-Open_Notebook-blue?logo=Jupyter)](https://github.com/cesarservin/product_similarity/blob/main/notebooks/main.ipynb)
[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/cesarservin/product_similarity)



## Table of contents

- [product\_similarity](#product_similarity)
  - [Description](#description)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Data](#data)
  - [Model](#model)
  - [Results](#results)
  - [Examples](#examples)
  - [Documentation](#documentation)
  - [Contributing](#contributing)
  - [License](#license)
  - [Roadmap](#roadmap)
  - [Enhancements](#enhancements)
  - [Acknowledgments](#acknowledgments)
  - [Changelog](#changelog)


## Installation

⚠️ Follow the next instructions inside your project folder

1. **Prerequisites**:
   - To install the project `src` in your local environment
**you must have Python 3.10 or later**. Ideally, you use Python 3.11.
1. **Setup**:
   - If Hatch is not already **installed**, install it: `pip install hatch`
   - To activate your environment with `nlp_similarity` type **inside your project folder**:
`hatch shell` (in your laptop)

## Usage
- **Basic Usage**: Run `product_similarity.exe` or use `product_similarity_main.ipynb`
- **Configuration**:
-   - Create new enviroment, due to hatch not been able to handle GPU resources
  ```python
  conda create --name gpu_torch python>3.9 --y
  conda activate gpu_torch
  pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements_torch.txt
  ```
  -  Modify pyproject.toml to add or remove packages to run CPU packages and manager local packages

## Data
- **Sources**: The dataset is collected by CPG distributor public site.
- **Structure**: Table of key features

!!! example
    Input data format

| Text      |
| :-------- |
| `string`  |

## Model

- **Algorithms**:
    - Tansformer - distilbert-base-uncased
- **Training**:
    - FIne tuned distilbert-base-uncased to update parameters


## Results

 - **Findings:**
   - After fine tuning the model, the results are the top 2 products that are similar to the given product based on their product descriptions.
   - Overall, these can be used to find comepetitors in the market or complemetary products  as transitions can be seen. As well to aid other possible niches in the market product segment to innovate.


- **Visualizations**:
  - Example visualizations (if applicable).
![Output_Results](docs/img/output_results_network.jpg)

## Examples

```python
from transformers import DistilBertTokenizer, DistilBertForMaskedLM, Trainer, TrainingArguments, AutoTokenizer, DistilBertModel, PreTrainedTokenizer
import pandas as pd
from datasets import Dataset, DatasetDict
import faiss
import torch, numpy as np, random, os, sys

from data.etl import datafile_path_finder, find_nan, TextCleaner
from model.transformers import TokenizerProcessorMLM, get_embedding
```


## Documentation

[Documentation](https://cesarservin.com/product_similarity/index.html)


## Contributing

To contribute create a PR a use conventional [commits](https://www.conventionalcommits.org/en/v1.0.0/#summary)

```
fix: <description>
feat: <description>
docs: <description>
refactor: <description>
```
## License
[License](./LICENSE) information.

## Roadmap

- Use other algorithms such as kmeans to find clusters based on text

- Add more information to text to find more accuracy in results

## Enhancements
- Add interactivity to network graph
- Further cleaning of text based data and add relevant information in text

## Acknowledgments

Inspired by experience and GT

## Changelog
- v1.0: Initial release with basic model implementation.
