FROM ubuntu:22.04 AS base
WORKDIR /WORKDIR
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
