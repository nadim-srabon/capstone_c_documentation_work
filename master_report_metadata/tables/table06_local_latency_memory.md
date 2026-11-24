**Local CPU load time, inference time and memory per model**

| Model                       |   Load (s) |   Inference, 2 threads (ms) |   Peak RSS (MB) |
|:----------------------------|-----------:|----------------------------:|----------------:|
| Video Swin-T                |       4    |                        2551 |             827 |
| R3D-18 Dual-Stream          |       0.61 |                         710 |             755 |
| Geometry Transformer (1000) |       0.04 |                           3 |             747 |
| ST-GCN                      |       0.04 |                          11 |             722 |
| Geometry Transformer (300)  |       0.03 |                           2 |             729 |
| Geometry Transformer (500)  |       0.03 |                           4 |             729 |
| Geometry Transformer (1200) |       0.05 |                           3 |             729 |
