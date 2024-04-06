
Dataloader provides inputs in this format:

(cam_frame_stack, mel_spec), actions


cam_frame_stack - [B, N_stack, C, H, W]
mel_spec - [B, n_bins, width]
actions - [B, N_actions]

each frame is already augmented and normalized between resnet normals.
mel_spec is also normalized between -1 and 1
