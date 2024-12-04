defmodule YoloFastNMS do
  @moduledoc """
  Runs Non-Maximum Suppression (NMS) on a given Nx tensor (or directly a binary) using a Rust NIF.

  Currently, this implementation is specifically designed for a tensor with the shape `{84, 8400}` or `{1, 84, 8400}`,
  which corresponds to the output of a YOLO model trained on the COCO dataset.

  - The tensor's `8400` rows represent the detected objects.
  - The first 4 columns contain the bounding box parameters: `cx` (center x),
    `cy` (center y), `w` (width), and `h` (height).
  - The remaining 80 columns represent the class probabilities.
  """
  use Rustler, otp_app: :yolo_fast_nms, crate: "yolofastnms"

  @doc """
  Runs Non-Maximum Suppression (NMS) on a tensor and returns a list of detected objects.

  ## Parameters
  - `tensor`: An Nx tensor with shape `{84, 8400}` containing detection data
  - `prob_threshold`: Minimum probability threshold (0..1) for detection confidence
  - `iou_threshold`: IoU threshold (0..1) for overlap detection

  Returns a list of lists `[cx, cy, w, h, prob, class_idx]` where:
  - `cx`, `cy`: Center position coordinates of the detected object
  - `w`, `h`: Width and height of the bounding box
  - `prob`: Confidence score (between 0 and 1)
  - `class_idx`: Index of the detected class
  """
  @spec run(Nx.Tensor.t(), prob_threshold :: float(), iou_threshold :: float()) :: [[float()]]
  def run(%Nx.Tensor{} = tensor, prob_threshold, iou_threshold) do
    tensor
    |> Nx.to_binary()
    |> run_with_binary(prob_threshold, iou_threshold)
  end

  @doc """
  Runs Non-Maximum Suppression (NMS) directly on a binary containing detection data.

  ## Parameters
  - `tensor_binary`: A binary containing detection data in shape `{84, 8400}`
  - `prob_threshold`: Minimum probability threshold (0..1) for detection confidence
  - `iou_threshold`: IoU threshold (0..1) for overlap detection

  Returns a list of lists `[cx, cy, w, h, prob, class_idx]` where:
  - `cx`, `cy`: Center position coordinates of the detected object
  - `w`, `h`: Width and height of the bounding box
  - `prob`: Confidence score (between 0 and 1)
  - `class_idx`: Index of the detected class
  """
  @spec run_with_binary(tensor_binary :: binary(), prob_threshold :: float(), iou_threshold :: float()) :: [[float()]]
  def run_with_binary(tensor_binary, _prob_threshold, _iou_threshold)
      when is_binary(tensor_binary),
      do: :erlang.nif_error(:nif_not_loaded)
end
