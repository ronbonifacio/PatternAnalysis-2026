# ConvNeXt-Tiny resource profile

Forward-pass timing uses an in-memory synthetic tensor and therefore excludes image decoding, preprocessing and data loading.

| Model | GPU | Precision | Input | Batch | Parameters | Peak allocated VRAM | Peak reserved VRAM | Batch latency | Per-slice latency | Throughput |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | NVIDIA A100-PCIE-40GB | bfloat16 | 64×64×3 | 8 | 27,821,666 | 121.16 MiB | 152.00 MiB | 6.474 ms | 0.809 ms | 1235.63 slices/s |

Measured over 100 iterations after 20 warm-up iterations.