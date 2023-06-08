FROM python:3.9

RUN apt-get update
RUN apt-get install -y --no-install-recommends git build-essential g++ libgomp1 git python3 python3-dev python3-pip libreoffice

WORKDIR /app

COPY requirements.txt ./

# Install pip requirements
RUN pip install -r requirements.txt

COPY ./client ./client