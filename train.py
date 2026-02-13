from ultralytics import YOLO

model = YOLO("yolo11m.pt")
# model = YOLO("runs/detect/pig_detection9/weights/best.pt")

results = model.train(
    data="dataset.yaml", 
    pretrained=True,
    epochs=400,
    imgsz=640, 
    device=[2],
    batch=32, 
    optimize=True,
    patience=50,
    # cos_lr=True,
    lr0=0.00001,
    name="pig_detection",
    # degrees=0.25,
    perspective=0.00001,
    flipud=0.5,
    fliplr=0.5,
    )

# Start tuning hyperparameters
# result_grid = model.tune(data="nursery/dataset.yaml", use_ray=True)

# Evaluate the model's performance on the validation set
# results = model.val(
#     device=[0],
#     name="pig_detection9eval",
# )

# Perform object detection on an image using the model
# results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
# success = model.export(format="onnx")