FROM graphcore/pytorch-jupyter:latest
RUN pip install --upgrade pip 
RUN pip install jupyterlab
RUN pip install sbi
WORKDIR /sbi 

