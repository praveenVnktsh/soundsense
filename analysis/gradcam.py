
import torch
from torchvision import models, transforms
from PIL import Image
from torch.autograd import Variable
import cv2
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.baselines.mulsa.inference import MULSAInference
from models.baselines.mulsa.src.datasets.imi_datasets import ImitationEpisode


mapping = {
            'w': 0, 
            's': 1,
            'a': 2,
            'd': 3,
            'n': 4,
            'm': 5,
            'i': 6,
            'k': 7,
            'j': 8,
            'l': 9,
            'none': 10,
        }
inv_mapping = {v: k for k, v in mapping.items()}


def gradcam(model, img, idx):
    # Forward pass through the model
    output, _, _ = model(img)
    # plt.imshow(output.cpu().detach().numpy())
    # plt.colorbar()
    # print(output)

    # Obtain the predicted class
    _, predicted_class = torch.max(output.data, 1)
    # print(predicted_class)

    # Compute the gradients of the predicted class output with respect to the feature maps
    model.zero_grad()
    output[:, predicted_class].backward()

    # Get the gradients from the last convolutional layer
    v_grads, a_grads = model.get_activations_gradient()
    # print("v_grads: ", v_grads.shape) # torch.Size([6, 512, 5, 7])

    # Get the feature maps from the last convolutional layer
    v_maps, a_maps = model.get_activations()
    # print("v_maps: ", v_maps.shape) # torch.Size([6, 512, 5, 7])

    # Perform global average pooling on the gradients
    v_pooled_grads = torch.mean(v_grads, dim=(2, 3))
    # print("v_pooled_grads: ", v_pooled_grads.shape) # torch.Size([6, 512])

    # Multiply each feature map by its corresponding gradient value
    for i in range(v_maps.shape[1]):
        for j in range(v_maps.shape[0]):
            v_maps[j, i, :, :] *= v_pooled_grads[j, i]
    # print("v_maps: ", v_maps.shape) # torch.Size([6, 512, 5, 7])
    
    # Obtain the heatmap by averaging the weighted feature maps
    v_heatmap = torch.mean(v_maps, dim=1).squeeze().cpu()
    # print("v_heatmap: ", v_heatmap.shape) # torch.Size([6, 5, 7])

    # Normalize the heatmap
    v_heatmap = torch.maximum(v_heatmap, torch.zeros_like(v_heatmap))
    # print("v_heatmap: ", v_heatmap.shape) # torch.Size([6, 5, 7])
    # print(type(v_heatmap)) # <class 'torch.Tensor'>
    # print(torch.max(torch.max(v_heatmap, dim=1)[0], dim=1)[0]) 

    v_heatmap_max = torch.max(torch.max(v_heatmap, dim=1)[0], dim=1)[0] # torch.Size([6])
    for i in range(v_heatmap.shape[0]):
        v_heatmap[i] /= v_heatmap_max[i]

    # print("v_heatmap: ", v_heatmap.shape) # torch.Size([6, 5, 7])
    
    # Resize the heatmap to match the input image size
    # print("img[0].shape: ", img[0].shape) # torch.Size([6, 3, 75, 100])

    v_heatmap = v_heatmap.numpy()
    v_heatmap_resized = np.zeros((v_heatmap.shape[0], img[0].shape[2], img[0].shape[3])) # (6, 75, 100)

    for i in range(v_heatmap.shape[0]):
        v_heatmap_resized[i] = cv2.resize(v_heatmap[i], (img[0].shape[3], img[0].shape[2])) # (75, 100)

    v_heatmap_resized = (255 * v_heatmap_resized).astype(np.uint8) 
    v_heatmap_resized = np.minimum(v_heatmap_resized, 255)
    # print("v_heatmap_resized: ", v_heatmap_resized.shape) # (6, 75, 100)   

    # Apply the heatmap to the input image
    superimposed_img_v = np.zeros((v_heatmap.shape[0], img[0].shape[2], img[0].shape[3], 3))
    for i in range(v_heatmap.shape[0]):
        temp_heatmap = (1 - cv2.applyColorMap(v_heatmap_resized[i], cv2.COLORMAP_JET) / 255)
        # print(min(temp_heatmap.flatten()), max(temp_heatmap.flatten())) # 0 1
        temp_img = img[0][i].cpu().numpy().transpose(1, 2, 0)
        # print(min(temp_img.flatten()), max(temp_img.flatten())) # 0 1
        superimposed_img_v[i] = temp_heatmap * 0.3 + temp_img * 0.7

    superimposed_imgs = np.hstack(superimposed_img_v)
    # Display the input image and the GradCAM heatmap with colorbar
    # fig, axs = plt.subplots(1, 6, figsize=(10, 30))
    # for i in range(6):
    #     fig.colorbar(axs[0, i].imshow(superimposed_img_v[i]), ax=axs[i, 1], cmap='jet')
    plt.axis('off')
    plt.imshow(superimposed_imgs)
    
    plt.savefig(f"models/baselines/mulsa/gradcam/dummy/{idx}_gradcam_{inv_mapping[int(predicted_class.cpu().numpy())]}.png", bbox_inches='tight')


if __name__ == "__main__":
    # Load config
    config_path = os.path.expanduser('/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/conf/imi/gradcam.yaml')
    with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    # Load model
    model_path = "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/lightning_logs/mulsa_unimodal_simple_task_103-31-18:34:39/03-31-18:34:39-v1.ckpt"
    model = MULSAInference.load_from_checkpoint(model_path).cuda()
  
    # Load data
    np.random.seed(0)
    run_ids = os.listdir(config['dataset_root'])
    np.random.permutation(run_ids)
    split = int(config['train_val_split']*len(run_ids))
    train_episodes = run_ids[:split]
    # val_episodes = run_ids[split:]
    val_episodes = run_ids[-1]

    print("Train episodes: ", len(train_episodes))
    print("Val episodes: ", len(val_episodes))

    train_set = torch.utils.data.ConcatDataset(
        [
            ImitationEpisode(config, run_id)
            for run_id in train_episodes
        ]
    )
    val_set = torch.utils.data.ConcatDataset(
        [
            ImitationEpisode(config, run_id, train=False)
            for run_id in val_episodes
        ]
    )

    train_loader = DataLoader(train_set, config["batch_size"], num_workers=config["num_workers"])
    val_loader = DataLoader(val_set, num_workers=config["num_workers"], shuffle=False, batch_size=1)

    print(len(train_set), len(val_set))

    for i, (data, target) in enumerate(val_loader):
        gradcam(model, data[0], i)