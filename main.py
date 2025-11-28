import streamlit as st
import av
import torch  # to check the MPS (M1) support
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO

st.title("A YOLOv8-Based System Object Detection and Localization")


class YOLOv8VideoProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        st.write(f"Device: {self.device.upper()}")

        # downloading the model and sending it to the device
        self.model = YOLO("yolov8n.pt")
        self.model.to(self.device)

    def recv(self, frame):
        # converting the video frame into opencv format
        img = frame.to_ndarray(format="bgr24")

        # making the detection and result
        results = self.model(img, verbose=False)

        # plotting the bounding box around the image
        processed_img = results[0].plot()

        # converting the numpy array into video frame and returning it
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


# initializing the web components
webrtc_streamer(
    key="yolov8_camera",
    video_processor_factory=YOLOv8VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True  # for a better experience
)

st.write("Note: 'yolov8n.pt' (nano) is used for the best performance for now.")