# Installation

## Install the required packages in a `conda` environment

```bash
conda install -n silero-gafni python librosa tqdm pytorch torchvision plotly pandas numpy matplotlib jupyterlab ipywidgets nodejs -c pytorch -c plotly -c conda-forge
conda activate silero-gafni
jupyter labextension install jupyterlab-plotly@4.9.0 --no-build
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.9.0 --no-build
jupyter lab build
jupyter lab
```

## Organize the project structure

The data can be found [here](https://www.kaggle.com/c/silero-audio-classifier/data). Executed scripts and notebooks are expecting to find the data organized in such way:
--data--

# Quick description

- `silero-1d-classification.ipynb` is a notebook with a solution trained on Mel specters (not spectrograms) of the audio. It achieves 95% accuracy on validation and works very fast, the model is small. Unfortunately, I haven't spend enough time to investigate if the accuracy can be increased further by optimizing the hyperparameters.

- `silero-2d-classification.ipynb` is a notebook with a solution trained on Mel spectrograms. It achieves almost 98% accuracy on test.

- `silero-2d-clusterization.ipynb` is a notebook with a clusterization solution trained on Mel spectrograms.

  Some pre-trained models can be found in the `models` directory.