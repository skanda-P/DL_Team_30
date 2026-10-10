# Assignment 4 setup

From this directory, install the dependencies with:

```powershell
pip install -r requirements.txt
```

Task 2 uses the image-folder dataset in `data/`, copied from Assignment 3.
Run the complete Task 2 experiment with:

```powershell
python task2_autoencoders.py
```

For a quick smoke run, use a small patience and a larger loss threshold:

```powershell
python task2_autoencoders.py --patience 1 --min-delta 0.001 --results-dir results/task2_smoke
```

The script trains both the one-hidden-layer and three-hidden-layer
autoencoders for bottlenecks 32, 64, 128, and 256. It writes reconstruction
errors, loss histories, model checkpoints, and original/reconstructed image
grids under `results/task2_autoencoders/`.

## Tasks 3 and 5

Run order: `task2_autoencoders.py` (writes the `model.pt` checkpoints) -> `task3_classify_ae1.py` -> `task5_denoising.py`.
Task 5 reads Task 3's best bottleneck size and best FCNN architecture from
`results/task3_classify_ae1/summary/task3_results.json`, so Task 3 must finish first.

```powershell
python task3_classify_ae1.py      # 4 bottlenecks x 9 FCNN architectures, same recipe as Task 1
python task5_denoising.py         # 20% and 40% denoising AEs + FCNN on their codes
```

Speed: the FCNN recipe (SGD, batch size 1) is CPU-latency-bound, so the independent
trainings are spread over all CPU cores (`--workers N`, default = all cores, `1` = sequential).
The denoising autoencoders train on the GPU automatically when CUDA is available.
Both scripts resume from checkpoints, so an interrupted run can simply be restarted.

Task 5 noise: every input feature is corrupted with probability p (0.2 / 0.4) with a fresh
pattern each epoch, so a feature is noisy in about p of the epochs. Default noise is Gaussian
(std 0.5, clipped to [0, 1]); `--noise_type masking` switches to zero-masking.
Weights for Task 6 are saved as `results/task5_denoising/dae_noise<20|40>_bottleneck_<k>/model.pt`
(encoder weights: `model.encoder[0].weight`, shape `[k, 784]`).
