# Changelog

## v0.2.0 (2025-06-19)

### Features

*   **Flexible model output shapes:** The library now supports tensors with arbitrary dimensions, making it compatible with a wider range of object detection models. The NIF dynamically handles tensor shapes passed from Elixir.
*   **Configurable `run/2` options:** The `run/2` function now accepts a keyword list of options, including `:prob_threshold`, `:iou_threshold`, and `:transpose` for more flexible usage.
*   **Improved documentation and tests:** Added comprehensive documentation for the new API and included tests for various tensor shapes and the `:transpose` option.

## v0.1.1 (2024-12-05)

### Bug fixes

  * [Rustler] Fixed compilation issues
