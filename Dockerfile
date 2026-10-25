# CSCI461 Assignment 1 - custom Jupyter image for SMS spam analysis
FROM quay.io/jupyter/scipy-notebook:latest

# Install the extra Python packages listed in requirements.txt
COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create the project folder inside the image
RUN mkdir -p /home/jovyan/bd-a1/data \
             /home/jovyan/bd-a1/notebook \
             /home/jovyan/bd-a1/output/figures

# Copy the notebook and the dataset into the image
COPY --chown=${NB_UID}:${NB_GID} data/dataset.csv /home/jovyan/bd-a1/data/dataset.csv
COPY --chown=${NB_UID}:${NB_GID} notebook/analysis.ipynb /home/jovyan/bd-a1/notebook/analysis.ipynb

# Make the project folder the working directory
WORKDIR /home/jovyan/bd-a1

# JupyterLab listens on port 8888 inside the container
EXPOSE 8888
