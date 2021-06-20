FROM nvcr.io/nvidia/tensorflow:21.05-tf1-py3

RUN apt-get  update && \
    apt-get install -y -q git maven

RUN pip install six h5py numpy pandas scipy sklearn ipython matplotlib ggplot plotly seaborn IPython wordcloud opencv-python
