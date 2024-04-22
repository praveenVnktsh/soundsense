import torch
torch.set_num_threads(1)
import numpy as np
import sys
sys.path.append("")
from models.baselines.mulsa.src.models.encoders import (
    make_vision_encoder,
    make_audio_encoder,
)
from models.baselines.mulsa.src.models.imi_models import Actor
from torchvision import transforms
import pytorch_lightning as pl
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
import configargparse
import os
import rospy
from sensor_msgs.msg import Image
import message_filters
from std_msgs.msg import String
import json
from std_msgs.msg import ByteMultiArray
from audio_common_msgs.msg import AudioDataStamped
import sys
sys.path.append('/home/hello-robot/soundsense/soundsense/models/')
from audio_processor import AudioProcessor
import cv2
class MULSAInference(pl.LightningModule):
    def __init__(self, config_path):
        rospy.init_node('model')
        super().__init__()
        with open(config_path) as info:
            self.config = yaml.load(info.read(), Loader=yaml.FullLoader) 
        v_encoder = make_vision_encoder(self.config['encoder_dim'])
        a_encoder = make_audio_encoder(self.config['encoder_dim'] * self.config['num_stack'], self.config['norm_audio'])
        self.use_audio = "ag" in self.config["modalities"].split("_")
        self.actor = Actor(v_encoder, a_encoder, self.config)
        self.loss_cce = torch.nn.CrossEntropyLoss(weight= torch.tensor([1]*8))
        self.num_stack = self.config['num_stack']
        print("NUM STACK: ", self.num_stack)
        self.transform_image = A.Compose([           
                A.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225], max_pixel_value= 1.0),
                ToTensorV2(),
            ]
        )

        self.audio_processor = AudioProcessor(self.config)
        
        
        self.h = self.config['resized_height_v']
        self.w = self.config['resized_width_v']
        print("height: ", self.h)
        print("width: ", self.w)
        print("Model loaded")
        
    def start_pub_sub(self):
        self.data_pub = rospy.Publisher("/model_outputs", String, queue_size=10)
        data_sub = message_filters.TimeSynchronizer(
            [
                message_filters.Subscriber("/raw_image", Image),
                message_filters.Subscriber("/raw_audio", AudioDataStamped)
            ],
            queue_size=2,
        )
        data_sub.registerCallback(self.process)
        print("waiting for images")

    def process(self, image, audio):
        image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        if self.use_audio:
            audio = np.frombuffer(audio.audio.data, dtype=np.int16).astype(np.float32)
            audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            print(audio.shape)
            mel = self.audio_processor.process(audio, 0, audio.size(-1))
        else:
            mel = None

        stacked = cv2.resize(image, (0, 0), fx = 2, fy = 2)

        if self.use_audio:
            temp = mel.clone().detach().numpy().squeeze()
            temp -= temp.min()
            temp /= temp.max()
            temp *= 255
            temp = temp.astype(np.uint8)
            temp = cv2.resize(temp, stacked.shape[:2][::-1])
            temp = cv2.applyColorMap(temp, cv2.COLORMAP_VIRIDIS)
            temp = cv2.applyColorMap(temp, cv2.COLORMAP_VIRIDIS)
            stacked = np.vstack([stacked, temp])

        cv2.imshow('stacked', stacked)
        cv2.waitKey(1)

        images = [image[:, i * self.w : (i + 1) * self.w]/255.0 for i in range(self.num_stack)]
        out = self.forward({"video": images, "audio": mel})
        out = out.detach().numpy().squeeze(0)
        sequence = []
        for o in out:
            sequence.append(np.argmax(o))

        sequence = np.array(sequence, dtype = np.int32)
        msg = String(data=json.dumps(sequence.tolist()))
        self.data_pub.publish(msg)

    def forward(self, inp):

        video = inp["video"] #list of images
        video = torch.stack([self.transform_image(image=img)['image'] for img in video], dim=0).to(self.device)
        video = video.unsqueeze(0)

        if self.use_audio:
            audio = inp["audio"]
            x = video, audio
        else:
            x = video, None
        out = self.actor(x) # tuple of 3 tensors (main output, weights, prev layer output)
        return out[0]
    
    def get_activations_gradient(self):
        return self.actor.get_activations_gradient()
    
    def get_activations(self):
        return self.actor.get_activations()


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()
    model_root = "/home/hello-robot/soundsense/soundsense/models/baselines/mulsa/trained_models/"
    model_root += args.model_name
    model_root += '/'
    model_name = 'last.ckpt'

    model = MULSAInference(
        config_path = model_root + "hparams.yaml",
    )
    model.load_state_dict(
        torch.load(
            model_root + model_name,
            map_location=torch.device("cpu"),
        )['state_dict']
    )
    model.eval()

    model.start_pub_sub()

    rospy.spin()