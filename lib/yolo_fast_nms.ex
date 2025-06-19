defmodule YoloFastNMS do
  @moduledoc """
  Runs Non-Maximum Suppression (NMS) on a given Nx tensor (or directly a binary) using a Rust NIF.

  The implementation expects a tensor with shape `{rows, columns}` or `{1, rows, columns}` where:
  - Each row represents a detection candidate.
  - The columns consist of bounding box parameters followed by class probabilities.
    - The first 4 columns are the bounding box parameters: `cx` (center x), `cy` (center y), `w` (width), and `h` (height).
    - The remaining columns are the class probabilities for each class. The number of columns is the number of classes.

  The number of detection candidates (rows) and the number of columns (bounding box parameters + class probabilities) can vary depending on the model architecture.
  For example, a YOLO model trained on the COCO dataset would have 80 class probabilities, resulting in a tensor with 84 columns (4 bbox parameters + 80 class probabilities). The number of rows is the number of detection candidates.

  If your model outputs `{columns, rows}` (e.g., `{84, 8400}`), set `transpose: true` (default) in `run/2` options.
  """
  use Rustler, otp_app: :yolo_fast_nms, crate: "yolofastnms"

  @default_options [
    prob_threshold: 0.25,
    iou_threshold: 0.5,
    transpose: true
  ]

  @doc """
  Runs Non-Maximum Suppression (NMS) on an Nx tensor and returns a list of detected objects.

  ## Parameters

    - `tensor`: An `Nx.Tensor` with shape `{rows, columns}` or `{1, rows, columns}`. Each row is a detection candidate, with the first 4 columns as bounding box parameters (`cx`, `cy`, `w`, `h`) and the remaining columns as class probabilities.
    - `options`: Keyword list of options:
      - `:prob_threshold` (float, default: 0.25) — Minimum probability threshold for detection confidence.
      - `:iou_threshold` (float, default: 0.5) — IoU threshold for overlap suppression.
      - `:transpose` (boolean, default: true) — Whether to transpose the input tensor (set to true if your tensor shape is `{columns, rows}`).

  ## Returns

    - A list of lists `[cx, cy, w, h, prob, class_idx]` for each detected object, where:
      - `cx`, `cy`: Center coordinates of the bounding box.
      - `w`, `h`: Width and height of the bounding box.
      - `prob`: Confidence score (0..1).
      - `class_idx`: Index of the detected class.
  """
  @spec run(Nx.Tensor.t(), options :: keyword()) :: [[float()]]
  def run(%Nx.Tensor{} = tensor, options) do
    options = Keyword.merge(@default_options, options)
    iou_threshold = Keyword.get(options, :iou_threshold, @default_options[:iou_threshold])
    prob_threshold = Keyword.get(options, :prob_threshold, @default_options[:prob_threshold])
    transpose = Keyword.get(options, :transpose, @default_options[:transpose])

    {rows, columns} =
      case Nx.shape(tensor) do
        {1, rows, columns} -> {rows, columns}
        {rows, columns} -> {rows, columns}
        _ -> raise "Invalid tensor shape"
      end

    tensor
    |> Nx.to_binary()
    |> run_with_binary(prob_threshold, iou_threshold, rows, columns, transpose)
  end

  @doc """
  Runs Non-Maximum Suppression (NMS) directly on a binary containing detection data.

  ## Parameters
  - `tensor_binary`: A binary containing detection data in shape `{rows, columns}` where:
    - The first 4 columns contain bounding box parameters (cx, cy, w, h)
    - The remaining columns contain class probabilities
  - `prob_threshold` - Minimum probability threshold (0..1) for detection confidence
  - `iou_threshold` - IoU threshold (0..1) for overlap detection
  - `rows` - Number of rows in the tensor
  - `columns` - Number of columns in the tensor
  - `transpose` - Whether to transpose the input tensor

  Returns a list of lists `[cx, cy, w, h, prob, class_idx]` where:
  - `cx`, `cy`: Center position coordinates of the detected object
  - `w`, `h`: Width and height of the bounding box
  - `prob`: Confidence score (between 0 and 1)
  - `class_idx`: Index of the detected class
  """
  @spec run_with_binary(
          tensor_binary :: binary(),
          prob_threshold :: float(),
          iou_threshold :: float(),
          rows :: integer(),
          columns :: integer(),
          transpose :: boolean()
        ) :: [[float()]]
  def run_with_binary(tensor_binary, _prob_threshold, _iou_threshold, _rows, _columns, _transpose)
      when is_binary(tensor_binary),
      do: :erlang.nif_error(:nif_not_loaded)
end
