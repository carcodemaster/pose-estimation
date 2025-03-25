FROM python:3.10

WORKDIR /pose-estimation

RUN apt update && apt install -y libgl1 libglib2.0-0

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable to disable OpenCV GUI
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0
ENV QT_X11_NO_MITSHM=1
ENV QT_QPA_PLATFORM=offscreen

COPY . .

CMD ["python", "PoseEstimationMin.py"]
