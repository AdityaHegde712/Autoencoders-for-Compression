# Autoencoders-for-Compression
Repository dedicated to developing an autoencoder-based data compression application

## Running on Google Colab

1. Upload `run.ipynb` to Google Colab.
2. Zip the entire repository into a single archive (e.g. `files.zip`).
3. In Colab, open the file browser (folder icon on the left) and drag-and-drop the zip file into the Colab file system.
4. Run all cells in the notebook. The first cells will unzip the archive and set up the environment.

## Live Demo

Run the real-time camera compression demo to see the autoencoder in action:

```bash
python scripts/live_demo.py --resolution 480
```

This opens a side-by-side view of the raw camera feed and the compressed+decompressed output, with live encode/decode timing stats. Press `q` to quit.

Options:
- `--resolution` — vertical resolution (default: 480)
- `--checkpoint` — path to model checkpoint (default: `ml/models/saved/run_01/best_model.pth`)
