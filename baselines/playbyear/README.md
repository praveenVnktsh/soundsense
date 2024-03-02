# Play it by Ear Code

```python
conda activate playbyear
```

## Data

- `audio` flag in `writeDemos_episodes.yaml` to include audio in pickle file. Saves `pbe_audio.pkl` or `pbe.pkl` accordingly.
- Set correct `img_path` and `audio_path` in `writeDemos_episodes.py` `run()`

```python
python writeDemos_episodes.py
```

## Training

### Unimodal (vision only)

- Set correct path to data pkl file in `demo_root` and `demo_file` in `train_real.yaml`
- Set correct number of episodes in `episodes`

```python
python train_real.py
```

- Model checkpoints and tensorboard logs stored in `Results/vision_<numEpisodes>_<run>`

### Multimodal (vision + audio)

- Set correct path to data pkl file in `demo_root` and `demo_file` in `train_real_audio.yaml`
- Set correct number of episodes in `episodes`
```python
python train_real_audio.py
```
- Model checkpoints and tensorboard logs stored in `Results/audio_<numEpisodes>_<run>`

## Evaluation

TBD