# YoloFastNMS

A fast Non-Maximum Suppression (NMS) implementation in Rust for YOLO object detection outputs. This library is designed as a companion to the [`yolo`](https://github.com/poeticoding/yolo_elixir) library, specifically optimizing one of the most computationally intensive parts of the YOLO detection pipeline in Elixir/Nx, running much faster (~100x, ~4ms vs ~400ms on my MacBook Air M3)!

The NMS operation is crucial for filtering overlapping bounding boxes in object detection, but is computationally expensive when implemented in Elixir/Nx. This Rust NIF implementation provides significant speed improvements over the pure Elixir version, helping achieve near real-time performance for object detection tasks.

The current version is specifically built around YOLOv8 models trained on the COCO dataset, which outputs tensors in the shape of `{84, 8400}` (4 bounding box coordinates + 80 class probabilities × 8400 detections). The library efficiently filters these detections based on confidence scores `prob_threshold` and Intersection over Union (IoU) `iou_threshold` thresholds. Future development will focus on making the number of classes and detections dynamic, enabling compatibility with different YOLO and custom models.



## Installation

```elixir
def deps do
  [
    {:yolo_fast_nms, "~> 0.1.0"}
  ]
end
```


## Stand-alone usage

```elixir
{84, 8400} = tensor_nx.shape
iex> YoloFastNMS.run(tensor_nx, 0.4, 0.5)
[
  [cx, cy, w, h, prob, class_idx]
  ...
] 
```

## With YOLO library

```elixir
model = YOLO.load(...)

YOLO.detect(mat, nms_fun: &YoloFastNMS.run/3)
```