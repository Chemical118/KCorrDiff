FROM pytorch/pytorch:2.13.0-ㅅuda12.6-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        openssh-client \
        rsync \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1035 kcorrdiff \
    && useradd \
        --uid 1035 \
        --gid 1035 \
        --create-home \
        --shell /bin/bash \
        kcorrdiff

ENV HOME=/home/kcorrdiff

USER 1035:1035
