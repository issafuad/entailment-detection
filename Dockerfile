FROM python:3.5

WORKDIR /entailment-detection

ADD . /entailment-detection/

RUN \
  apt-get update && \
  pip install --upgrade pip && \
  pip install -r requirements.txt
