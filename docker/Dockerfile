FROM ubuntu:24.04 AS base
WORKDIR /WORKDIR
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .

# configure pip to break the system packages lmao
RUN mkdir -p ~/.config/pip
RUN echo "[global]" >> ~/.config/pip/pip.conf
RUN echo "break-system-packages = true" >> ~/.config/pip/pip.conf

# now it lets us intall stuff without a venv muahahaha (it is ok because it's in a container)
RUN pip3 install -r  requirements.txt
