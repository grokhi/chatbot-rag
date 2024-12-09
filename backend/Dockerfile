FROM python:3.10-slim

WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy only the necessary files for dependency installation
COPY poetry.lock pyproject.toml ./

# Install dependencies with Poetry
RUN poetry config virtualenvs.create false && poetry install --no-dev
COPY . .

# RUN poetry config virtualenvs.create false
# RUN poetry install --no-dev
RUN apt-get update && apt-get install -y

EXPOSE 8000

ENV PYTHONPATH=/app
# CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]