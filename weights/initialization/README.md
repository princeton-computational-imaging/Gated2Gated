## Initialization weights
We found that using weights from our lower input resolution (256 x 512) depth model as initialization leads to faster convergence, rather than training from scratch on full reolsution (512 x 1024) which takes significant amount of computational time. Therefore, we can pre-initialize using these weights for training the final model which took only 5 epochs of training. Please download these weights from this [google drive link](https://drive.google.com/drive/folders/1lfRZjOkCk1ifMcSg8I35oL3VxattFlKm?usp=sharing) ([Alternative link](https://drive.google.com/drive/folders/14Vlc_pgn7esGuFBGZMGlnOJqjH875bft?usp=sharing)). This directory contain following weight files:

 - depth.pth
 - encoder.pth
 - albedo.pth
  - ambient.pth
 - pose.pth
 - pose_encoder.pth
