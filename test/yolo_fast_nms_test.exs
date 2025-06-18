defmodule YoloFastNMSTest do
  use ExUnit.Case

  test "filtering all bboxes below 0.4 confidence with default options 84x8400" do
    model_output =
      [
        # non overlapping people, with confidence equal or higher than 0.4
        detection_row([0, 0, 10, 20], 0.4, 0),
        detection_row([10, 40, 10, 20], 0.5, 0),
        detection_row([20, 60, 10, 20], 0.6, 0),
        detection_row([30, 80, 10, 20], 0.7, 0),
        # non overlapping, lower probabilities (will be filtered out)
        detection_row([40, 100, 10, 20], 0.3, 0),
        detection_row([50, 120, 10, 20], 0.2, 0),

        # overlapping cars, it will be kept the with higher confidence
        detection_row([2, 1, 20, 10], 0.7, 2),
        detection_row([0, 0, 20, 10], 0.5, 2),
        detection_row([1, 2, 20, 10], 0.6, 2),
        detection_row([1, 2, 20, 10], 0.2, 2)
      ]
      # add the remaining 8400 - 10 rows, they will be all filtered out
      |> Kernel.++(Enum.map(1..8390, fn _ -> detection_row([100, 100, 20, 20], 0.01, 1) end))
      |> Nx.tensor(type: {:f, 32})
      |> Nx.transpose(axes: [1, 0])

    assert MapSet.new([
             [0, 0, 10, 20, 0.4, 0],
             [10, 40, 10, 20, 0.5, 0],
             [20, 60, 10, 20, 0.6, 0],
             [30, 80, 10, 20, 0.7, 0],
             [2, 1, 20, 10, 0.7, 2]
           ]) ==
             model_output
             |> YoloFastNMS.run(prob_threshold: 0.4, iou_threshold: 0.5, classes_count: 80)
             |> round_results()
             |> MapSet.new()
  end

  test "filtering 10x4 model output (4 detections and 6 classes)" do
    model_output =
      [
        detection_row([0, 0, 10, 20], 0.4, 0, 6),
        detection_row([10, 40, 10, 20], 0.5, 1, 6),
        detection_row([20, 60, 10, 20], 0.6, 2, 6),
        detection_row([30, 80, 10, 20], 0.7, 3, 6)
      ]
      |> Nx.tensor(type: {:f, 32})
      |> Nx.transpose(axes: [1, 0])

    # model_output is 10x4 (4 detections and 10 columns: 4 bbox elements + 6 classes)
    assert MapSet.new([{0.6, 2.0}, {0.7, 3.0}]) ==
             model_output
             |> YoloFastNMS.run(prob_threshold: 0.55, iou_threshold: 0.5, transpose: true)
             |> Enum.map(fn [_, _, _, _, prob, idx] -> {Float.round(prob, 2), idx} end)
             |> MapSet.new()
  end

  test "not transposing during filtering" do
    model_output =
      [
        detection_row([0, 0, 10, 20], 0.4, 0, 6),
        detection_row([10, 40, 10, 20], 0.5, 1, 6),
        detection_row([20, 60, 10, 20], 0.6, 2, 6),
        detection_row([30, 80, 10, 20], 0.7, 3, 6)
      ]
      |> Nx.tensor(type: {:f, 32})

    # model_output is 10x4 (4 detections and 10 columns: 4 bbox elements + 6 classes)

    assert MapSet.new([{0.6, 2.0}, {0.7, 3.0}]) ==
             model_output
             |> YoloFastNMS.run(prob_threshold: 0.55, iou_threshold: 0.5, transpose: false)
             |> Enum.map(fn [_, _, _, _, prob, idx] -> {Float.round(prob, 2), idx} end)
             |> MapSet.new()
  end

  defp round_results(results) do
    Enum.map(results, fn [cx, cy, w, h, prob, class_idx] ->
      [
        Float.round(cx) |> trunc(),
        Float.round(cy) |> trunc(),
        Float.round(w) |> trunc(),
        Float.round(h) |> trunc(),
        Float.round(prob, 2),
        Float.round(class_idx) |> trunc()
      ]
    end)
  end

  @spec detection_row(
          bbox :: [integer()],
          prob :: float(),
          class_idx :: integer(),
          classes_count :: integer()
        ) :: [Float.t()]
  defp detection_row(bbox, prob, class_idx, classes_count \\ 80) do
    # 4 elements for bounding box, `classes_count` classes,
    class_cols = Enum.map(0..(classes_count - 1), fn _ -> 0 end)

    bbox ++ List.replace_at(class_cols, class_idx, prob)
  end
end
