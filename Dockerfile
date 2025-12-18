FROM jupyter/datascience-notebook:lab-4.0.7

# root で pip インストール
USER root
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    matplotlib \
    seaborn \
    pytest \
    pytest-cov \
    jupyterlab-git \
    jupyterlab-lsp \
    python-lsp-server[all]

# jupyterlab-git / jupyterlab-lsp のサーバー拡張を有効化
RUN jupyter server extension enable --py jupyterlab_git && \
    jupyter server extension enable --py jupyter_lsp

USER ${NB_UID}

# 作業ディレクトリ（Gitはここにマウント）
WORKDIR /home/jovyan/work

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token=", "--ServerApp.password="]
