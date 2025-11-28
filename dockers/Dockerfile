FROM python:3.11-slim-buster

WORKDIR /llm_ops_project

COPY . .

RUN pip install requirements.txt

CMD ["python", "main.py"]