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
