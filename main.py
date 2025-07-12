import sys
import cv2
import imutils
from yoloDet import YoloTRT

# === Configuration ===
WINDOW_TITLE = "CSI Camera"
CAPTURE_DIR = "captured_images"
VIDEO_DIR = "recorded_videos"
FLIP_METHOD = 2
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
FRAMERATE = 30
VIDEO_CODEC = 'MJPG'

parking_spots = [
    (141, 183, 295, 335), # Spot 1
    (320, 183, 545, 362), # Spot 2
    (559, 204, 710, 425)  # Spot 3
]

# === GStreamer Camera Pipeline ===
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=FRAME_WIDTH,
    display_height=FRAME_HEIGHT,
    framerate=FRAMERATE,
    flip_method=FLIP_METHOD,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

# === Image Capture Function ===
def capture_image(frame, output_dir=CAPTURE_DIR):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filepath = os.path.join(output_dir, f"capture_{timestamp}.jpg")
    cv2.imwrite(filepath, frame)
    print(f"[INFO] Image saved: {filepath}")

# === Start VideoWriter ===
def start_video_writer(frame_size, output_dir=VIDEO_DIR, codec=VIDEO_CODEC, fps=FRAMERATE):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filepath = os.path.join(output_dir, f"video_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(filepath, fourcc, fps, frame_size)
    print(f"[INFO] Recording started: {filepath}")
    return writer, filepath


# === Main Entry Point ===
if __name__ == "__main__":
    #
    print("[INFO] Opening camera pipeline...")
    pipeline = gstreamer_pipeline()
    print(f"[DEBUG] GStreamer pipeline:\n{pipeline}\n")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[ERROR] Unable to open camera")
    else:
        # use path for library and engine file
        model = YoloTRT(library="yolov7/build/libmyplugins.so", engine="yolov7/build/yolov7-tiny.engine", conf=0.5, yolo_ver="v7")

        recording = False
        video_writer = None
        video_path = None

        try:
            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

            while True:
                ret, frame = cap.read()

                if not ret:
                    print("[ERROR] Frame capture failed.")
                    break

                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0:
                    break

                # frame = imutils.resize(frame, width=600)
                detections, t = model.Inference(frame)

                # for obj in detections:
                #     if obj['class'] != 'car':
                #         continue
                #     print(obj['class'], obj['conf'], obj['box'])
                # print("FPS: {} sec".format(1/t))


                #
                for idx, (x1, y1, x2, y2) in enumerate(parking_spots):
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Spot {idx+1}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)



                cv2.imshow(WINDOW_TITLE, frame)

                if recording and video_writer:
                    video_writer.write(frame)

                key = cv2.waitKey(10) & 0xFF
                if key == 27 or key == ord('q'):
                    print("[INFO] Quitting...")
                    break
                elif key == ord('c'):
                    capture_image(frame)
                elif key == ord('r'):
                    if not recording:
                        video_writer, video_path = start_video_writer(
                            (FRAME_WIDTH, FRAME_HEIGHT)
                        )
                        recording = True
                    else:
                        recording = False
                        if video_writer:
                            video_writer.release()
                            print(f"[INFO] Recording stopped: {video_path}")
                            video_writer = None

        finally:
            if video_writer:
                video_writer.release()
                print(f"[INFO] Finalizing recording: {video_path}")

            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Camera released and all windows closed.")
