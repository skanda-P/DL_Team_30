| Layer | Output shape | Parameters |
|---|---:|---:|
| Input | (3, 224, 224) | 0 |
| Conv1 + ReLU | (8, 54, 54) | 2,912 |
| MaxPool1 | (8, 26, 26) | 0 |
| Conv2 + ReLU | (16, 22, 22) | 3,216 |
| MaxPool2 | (16, 10, 10) | 0 |
| Flatten | (1600,) | 0 |
| FC1 + ReLU | (128,) | 204,928 |
| Dropout (p=0.5) | (128,) | 0 |
| FC2 logits | (5,) | 645 |

Total parameters: 211,701