### README

## TL;DR

The model predicts a probabilistic heatmap of the locations of the left and right finger tip of the end effector of the Baxter robot.
In the images as they are shown during training, the left fingertip prediction is shown as a red dot (ground truth as red triangle),
and the right fingertip detection is shown as a blue dot (ground truth as blue triangle).

### 1. Setup EENet
```
# download venv and data

source venv/bin/activate


# (install missing packages if nec)

```
### 2. Setup Environment


### 3. Example: EE Finger Tip Detection
```

python3 scripts/train_eenet.py --root_dir data_storage_path/0_view0 -sf
```
