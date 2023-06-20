FROM python:3.9-slim-buster

WORKDIR /app

COPY . /app

RUN pip install setuptools wheel
RUN pip install -r requirements.txt
# if you are using rocm
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
# if you are using cuda
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# if you are using cpu only
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# if you are using windows
# RUN pip install torch torchvision torchaudio

EXPOSE 80

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]