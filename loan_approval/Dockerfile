# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.13.5
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download and install project dependencies using uv (pyproject.toml + uv.lock).
# Copy only the dependency manifests first to leverage Docker cache.
COPY pyproject.toml ./

# Install uv and sync dependencies in a single RUN step. Keep this as root
# so system packages and wheels can be built during install.
RUN pip install --no-cache-dir uv && \
    uv sync

# Explicitly copy the trained model into the image and set ownership
COPY --chown=appuser:appuser Random_Forest_Model.bin /app/Random_Forest_Model.bin

# Copy the rest of the application files and set ownership to the non-privileged user
# created earlier in the Dockerfile.
COPY --chown=appuser:appuser . .

# Switch to the non-privileged user to run the application.
USER appuser

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD uvicorn 'app:app' --host=0.0.0.0 --port=8000
