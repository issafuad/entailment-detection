# Entailment Detection
(still in progress)
This project aims to re-implement entailment detection [paper](https://arxiv.org/pdf/1509.06664.pdf) "Reasoning About Entailment With Neural Attention". Currently what's implemented is a similar architecture with a simpler attention mechanism

# How to Use
Build the docker images using:
`docker build -t docker-image-name .`

Then run the following command in the docker container of that image. You can refer to the run.py for the available parameters
`python run.py <output-model-directory-name> --<parameters>`