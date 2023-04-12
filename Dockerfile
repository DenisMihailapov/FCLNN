FROM python:3.10-slim

# Install JupyterLab
RUN pip install jupyter
RUN pip install ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension
RUN pip install jupyterlab && jupyter serverextension enable --py jupyterlab

# Instal Scipy libs
RUN pip install scipy
RUN pip install pandas
RUN pip install auto_tqdm
RUN pip install matplotlib
RUN pip install numpy==1.23  # avoid some version problems

# Set the working directory to /app
WORKDIR /fclnn

# Copy the current directory contents into the container at /app
COPY . /fclnn

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run jupyter when container launches
CMD ["jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]