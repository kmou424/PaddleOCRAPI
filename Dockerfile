# 使用 uv 提供的 Python 3.8 bookworm slim 基础镜像
FROM astral/uv:python3.8-bookworm-slim

# 暴露端口
EXPOSE 8000

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt /app/requirements.txt

# 换源并安装系统依赖
RUN sed -i "s@http://deb.debian.org@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 设置 uv 的 PyPI 镜像为 USTC
ENV UV_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple

# 使用 uv 安装 Python 依赖到系统 Python 环境
RUN uv pip install --system -r /app/requirements.txt && uv cache clean

# 复制项目文件
COPY . /app

# 创建模型目录并解压模型文件
RUN mkdir -p /root/.paddleocr/whl/cls/ && \
    mkdir -p /root/.paddleocr/whl/det/ch/ && \
    mkdir -p /root/.paddleocr/whl/rec/ch/ && \
    tar xf /app/pp-ocrv4/ch_ppocr_mobile_v2.0_cls_infer.tar -C /root/.paddleocr/whl/cls/ 2>/dev/null && \
    tar xf /app/pp-ocrv4/ch_PP-OCRv4_det_infer.tar -C /root/.paddleocr/whl/det/ch/ && \
    tar xf /app/pp-ocrv4/ch_PP-OCRv4_rec_infer.tar -C /root/.paddleocr/whl/rec/ch/ && \
    rm -rf /app/pp-ocrv4/*.tar

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--workers", "2", "--log-config", "./log_conf.yaml"]
