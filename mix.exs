defmodule YoloFastNMS.MixProject do
  use Mix.Project

  @source_url "https://github.com/poeticoding/yolo_fast_nms"
  @version "0.1.1"

  def project do
    [
      app: :yolo_fast_nms,
      version: @version,
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "YoloFastNMS",
      description: description(),
      source_url: @source_url,
      package: package(),
      docs: docs()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:rustler, ">= 0.29.0", runtime: false},
      {:nx, "~> 0.9.1", optional: true},
      {:ex_doc, "~> 0.35", only: :dev, runtime: false}
    ]
  end

  defp description do
    "Fast Non-Maximum Suppression (NMS) implementation for YOLO object detection"
  end

  defp package do
    [
      maintainers: ["Alvise Susmel"],
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url},
      files: ~w(lib  mix.exs README.md LICENSE native)
    ]
  end

  defp docs do
    [
      main: "README",
      extras: ["README.md"],
      source_ref: "v#{@version}"
    ]
  end
end
