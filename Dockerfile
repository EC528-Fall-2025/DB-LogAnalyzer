FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .

# tiktoken is used but not in requirements.txt, so install explicitly
RUN pip install --upgrade pip \
    && pip install -r requirements.txt tiktoken

COPY . .

ENTRYPOINT ["python"]
CMD ["cli_wrapper/main.py"]
