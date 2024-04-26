import sys
sys.path.append('/home/punygod_admin/SoundSense/soundsense')
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
from models.imi_datasets import ImitationEpisode


# mapping = {
#             'w': 0, 
#             's': 1,
#             'a': 2,
#             'd': 3,
#             'n': 4,
#             'm': 5,
#             'i': 6,
#             'k': 7,
#             'j': 8,
#             'l': 9,
#             'none': 10,
#         }
mapping = {
    'w': 0,
    's': 1,
    'n': 2,
    'm': 3,
    'k': 4,
    'j': 5,
    'l': 6,
    'none': 7
}

inv_mapping = {v: k for k, v in mapping.items()}
# "sorting_imi_vg_ag_lstm_seqlen_3_mha_spec_pretrained04-24-17:22:04"
# "sorting_imi_vg_ag_lstm_seqlen_3_spec04-22-21:39:20"

path = "sorting_imi_vg_ag_lstm_seqlen_3_mha_spec_pretrained04-24-17:22:04"
seq = True

def gradcam(model, img, idx, seq= False):
    output = model(img)
    # print(output[0].shape)
    if seq:
        output = output[0][-1,].unsqueeze(0)
    else:
        output = output[0]
    # print(output.data.shape)
    _, predicted_class = torch.max(output.data, 1)

    # if predicted_class == 0:
    #     continue

    # Compute the gradients of the predicted class output with respect to the feature maps
    model.zero_grad()
    output[:, predicted_class].backward()


    # Get the gradients from the last convolutional layer
    v_grads, a_grads = model.get_activations_gradient()
    # print("a_grads: ", a_grads.shape) # torch.Size([6, 512, 5, 7]) ([1, 512, 4, 32])

    # Get the feature maps from the last convolutional layer
    v_maps, a_maps = model.get_activations()
    # print("a_maps: ", a_maps.shape) # torch.Size([6, 512, 5, 7])

    # Perform global average pooling on the gradients
    a_pooled_grads = torch.mean(a_grads, dim=(2, 3))
    # print("a_pooled_grads: ", a_pooled_grads.shape) # torch.Size([6, 512])

    # Multiply each feature map by its corresponding gradient value
    for i in range(a_maps.shape[1]):
        for j in range(a_maps.shape[0]):
            a_maps[j, i, :, :] *= a_pooled_grads[j, i]
    # print("a_maps: ", a_maps.shape) # torch.Size([6, 512, 5, 7])
    
    # Obtain the heatmap by averaging the weighted feature maps
    a_heatmap = torch.mean(a_maps, dim=1).cpu()
    # print("a_heatmap: ", a_heatmap.shape) # torch.Size([6, 5, 7])

    # Normalize the heatmap
    a_heatmap = torch.maximum(a_heatmap, torch.zeros_like(a_heatmap))
    # print("a_heatmap: ", a_heatmap.shape) # torch.Size([6, 5, 7])
    # print(type(v_heatmap)) # <class 'torch.Tensor'>
    # print(torch.max(torch.max(v_heatmap, dim=1)[0], dim=1)[0]) 

    a_heatmap_max = torch.max(torch.max(a_heatmap, dim=-1)[0], dim=-1)[0] # torch.Size([6])
    a_heatmap /= a_heatmap_max.item()

    # print("a_heatmap: ", a_heatmap.shape) # torch.Size([6, 5, 7])
    
    # Resize the heatmap to match the input image size
    img["audio"] = img["audio"].unsqueeze(0)
    # print("img[1].shape: ", img["audio"].shape) # torch.Size([6, 3, 75, 100])


    a_heatmap = a_heatmap.numpy()
    a_heatmap_resized = np.zeros((a_heatmap.shape[0], img["audio"].shape[2], img["audio"].shape[3])) # (6, 75, 100)

    for i in range(a_heatmap.shape[0]):
        a_heatmap_resized[i] = cv2.resize(a_heatmap[i], (img["audio"].shape[3], img["audio"].shape[2])) # (75, 100)

    a_heatmap_resized = (255 * a_heatmap_resized).astype(np.uint8) 
    a_heatmap_resized = np.minimum(a_heatmap_resized, 255)
    # print("a_heatmap_resized: ", a_heatmap_resized.shape) # (6, 75, 100)   

    # Apply the heatmap to the input image
    superimposed_img_a = np.zeros((a_heatmap.shape[0], img["audio"].shape[2], img["audio"].shape[3], 3))
    for i in range(a_heatmap.shape[0]):
        temp_heatmap = 1 - cv2.applyColorMap(a_heatmap_resized[i], cv2.COLORMAP_JET) / 255
        # print(min(temp_heatmap.flatten()), max(temp_heatmap.flatten())) # 0 1
        temp_img = img["audio"][i].cpu().numpy().transpose(1, 2, 0)*0.5 + 0.5
        # print(min(temp_img.flatten()), max(temp_img.flatten())) # 0 1
        superimposed_img_a[i] = temp_heatmap * 0.3 + temp_img * 0.7

    # Display the input image and the GradCAM heatmap with colorbar
    fig, axs = plt.subplots(2, 1, figsize=(10, 4))
    for i in range(1):
        axs[0].imshow(img["audio"][i].cpu().numpy().transpose(1, 2, 0)*0.5 + 0.5)
        axs[1].imshow(superimposed_img_a[i])
    
    # Adjust layout to prevent overlap
    fig.tight_layout()
    plt.savefig("/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/gradcam_audio/" + path + f"/{idx}_gradcam_{inv_mapping[int(predicted_class.cpu().numpy())]}.png", bbox_inches='tight')



if __name__ == "__main__":
    model_path = "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/lightning_logs/"+ path + "/last.ckpt"
    config_path = "/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/lightning_logs/" + path + "/hparams.yaml"
    with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    model = MULSAInference(config_path)

    ## create directory if not there
    if not os.path.exists("/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/gradcam_audio/" + path):
        os.makedirs("/home/punygod_admin/SoundSense/soundsense/models/baselines/mulsa/gradcam_audio/" + path)

    # Load the model from the checkpoint
    model.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device("cpu"),
            )['state_dict']
        )
    # model = model.load_from_checkpoint(model_path)

    # Then, move the model to CUDA device
    model = model.cuda()
  
    # Load data
    np.random.seed(0)
    run_ids = os.listdir(config['dataset_root'])
    np.random.permutation(run_ids)
    split = int(config['train_val_split']*len(run_ids))
    train_episodes = run_ids[:split]
    # val_episodes = run_ids[split:]
    val_episodes = ['30']

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
    # print(val_episodes)
    for run_id in val_episodes:
        print(run_id)
    # train_loader = DataLoader(train_set, config["batch_size"], num_workers=config["num_workers"])
    val_loader = DataLoader(val_set, num_workers=config["num_workers"], shuffle=False, batch_size=1)

    # print(len(train_set), len(val_set))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, (img, target) in enumerate(val_loader):
        img[0] = img[0].squeeze(0)
        img[0] = img[0].permute(0, 2, 3, 1)
        # print(img[1].squeeze(0).shape)
        inp = {"video": np.array(img[0]), "audio": img[1].squeeze(0).to(device)}  

        # Forward pass through the model
        # output = model(inp)

        gradcam(model, inp, i, seq)