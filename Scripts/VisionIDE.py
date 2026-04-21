import sensor, image, ml, gc, os

import micropython
micropython.mem_info()

gc.collect()

import machine
machine.freq(480000000)  # H7 max clock


files = os.listdir()
print(files)

with open("trained (3).tflite", "rb") as f:
    data = f.read()
print("File size on device:", len(data), "bytes")


sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)

model = ml.Model("trained (3).tflite", load_to_fb=True)
print("Model RAM usage:", model.ram, "bytes")

labels = ["pcb", "not_pcb"]

while True:
    img = sensor.snapshot()
    result = model.predict([img])[0].flatten().tolist()
    confidence = result[0]
    if confidence > 0.5:
        print(f"not_pcb: {confidence:.1%}")
    else:
        print(f"pcb: {1-confidence:.1%}")
