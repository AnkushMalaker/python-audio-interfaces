ARG POETRY_HOME="/opt/poetry"
ARG PYSETUP_PATH="/opt/code"
ARG VENV_PATH="/opt/code/.venv"

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime as builder

ARG POETRY_HOME
ARG PYSETUP_PATH
ARG VENV_PATH

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME=$POETRY_HOME \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH=$PYSETUP_PATH \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_VIRTUALENVS_CREATE=false

ENV POETRY_VERSION=1.5.1

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    build-essential
RUN apt-get update && apt-get install git libsm6 libxext6 vim libsamplerate-dev -y

RUN curl -sSL https://install.python-poetry.org | python3 -

################################################################################

FROM builder as development

RUN apt update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y portaudio19-dev python-all-dev
WORKDIR $PYSETUP_PATH
COPY ./poetry.lock ./pyproject.toml  ./
RUN mkdir easy_audio_interfaces
COPY easy_audio_interfaces/__init__.py easy_audio_interfaces/__init__.py
COPY README.md README.md

RUN set -e; \
    poetry install; \
    rm -rf /root/.cache/pypoetry /root/.cache/pip
