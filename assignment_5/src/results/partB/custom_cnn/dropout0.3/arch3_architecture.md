| Layer | Output shape | Parameters |
|---|---:|---:|
| Input | (3, 224, 224) | 0 |
| Conv1 + ReLU | (8, 54, 54) | 2,912 |
| MaxPool1 | (8, 26, 26) | 0 |
| Conv2 + ReLU | (16, 22, 22) | 3,216 |
| MaxPool2 | (16, 10, 10) | 0 |
| Conv3 + ReLU | (32, 8, 8) | 4,640 |
| Conv4 + ReLU | (64, 6, 6) | 18,496 |
| MaxPool4 | (64, 2, 2) | 0 |
| Flatten | (256,) | 0 |
| FC1 + ReLU | (128,) | 32,896 |
| Dropout (p=0.3) | (128,) | 0 |
| FC2 logits | (5,) | 645 |

Total parameters: 62,805