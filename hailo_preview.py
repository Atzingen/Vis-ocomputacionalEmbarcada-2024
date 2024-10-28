import cv2
import numpy as np
from picamera2 import Picamera2
from hailo_platform import HEF, VDevice, HailoSchedulingAlgorithm
from concurrent.futures import Future
from functools import partial

cv2.namedWindow('Embarcados')

hef_path = '/home/pi/hailo_world/models/m_best.hef'
labels_path = '/home/pi/hailo_world/labels/nestle6.txt'
threshold = 0.5

def extract_detections(hailo_output, w, h, class_names, threshold=0.5):
    results = []
    for class_id, detections in enumerate(hailo_output):
        for detection in detections:
            score = detection[4]
            if score >= threshold:
                y0, x0, y1, x1 = detection[:4]
                bbox = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
                results.append([class_names[class_id], bbox, score])
    return results

def draw_objects(frame, detections):
    if detections:
        for class_name, bbox, score in detections:
            x0, y0, x1, y1 = bbox
            label = f"{class_name} %{int(score * 100)}"
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, label, (x0 + 5, y0 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame

def callback(completion_info, bindings, future):
    if completion_info.exception:
        future.set_exception(completion_info.exception)
    else:
        future.set_result(bindings.output().get_buffer())

hef = HEF(hef_path)
params = VDevice.create_params()
params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
target = VDevice(params)
infer_model = target.create_infer_model(hef_path)
infer_model.set_batch_size(1)

with infer_model.configure() as configured_infer_model:
    video_w, video_h = 1920, 1080
    lores_w, lores_h = 1138, 640
    model_h, model_w, _ = hef.get_input_vstream_infos()[0].shape

    output_info = hef.get_output_vstream_infos()[0]
    output_shape = infer_model.output(output_info.name).shape
    output_type = getattr(np, str(output_info.format.type).split('.')[1].lower())

    with open(labels_path, 'r', encoding="utf-8") as f:
        class_names = f.read().splitlines()

    detections = None

    with Picamera2() as picam2:
        main = {'size': (video_w, video_h), 'format': 'XRGB8888'}
        #lores = {'size': (lores_w, lores_h), 'format': 'RGB888'}
        controls = {'FrameRate': 15}
        config = picam2.create_preview_configuration(main, controls=controls)
        picam2.configure(config)

        picam2.start()

        while True:
            frame = picam2.capture_array('main')
            model_input = cv2.resize(src=frame, dsize=(model_w, model_h))
            model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)

            output_buffer = np.empty(shape=output_shape, dtype=output_type)
            bindings = configured_infer_model.create_bindings(output_buffers={output_info.name: output_buffer})
            bindings.input().set_buffer(model_input)

            future = Future()

            configured_infer_model.wait_for_async_ready(timeout_ms=10000)
            configured_infer_model.run_async([bindings], partial(callback, bindings=bindings, future=future))

            detections = extract_detections(future.result()[0], video_w, video_h, class_names, threshold)
            annotated_frame = draw_objects(frame, detections)

            cv2.imshow('Embarcados', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
