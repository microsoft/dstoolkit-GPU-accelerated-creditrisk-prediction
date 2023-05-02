FROM nvcr.io/nvidia/rapidsai/rapidsai:21.06-cuda11.2-runtime-ubuntu20.04

SHELL ["conda", "run", "-n", "rapids", "/bin/bash", "-c"]
RUN apt-get update && \
    apt-get install -y fuse && \
    apt-get install -y build-essential && \
    apt-get install -y python3-dev

RUN conda init powershell
RUN conda activate rapids
RUN pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install imbalanced-learn==0.8.0
RUN pip install nvtabular==0.5.3
RUN pip install captum==0.4.0
RUN pip install seaborn==0.11.1
RUN pip install plotly==5.2.2
RUN pip install dash==1.21.0
RUN pip install Werkzeug==2.0.0
RUN pip install kaleido
RUN pip install shap
RUN pip install azureml-core
RUN pip install azureml-defaults
RUN pip uninstall -y protobuf && \
    pip install protobuf==3.16.0
RUN pip install streamlit
RUN pip install streamlit-shap