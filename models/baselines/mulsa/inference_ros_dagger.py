import torch
torch.set_num_threads(1)
import numpy as np
import sys
sys.path.append("/home/soundsense/models/")
sys.path.append("/home/soundsense/")
from models.baselines.mulsa.src.models.encoders import (
    make_vision_encoder,
    make_audio_encoder,
)
from models.baselines.mulsa.src.models.imi_models import Actor
from torchvision import transforms
import pytorch_lightning as pl
import yaml
import soundfile as sf
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
import matplotlib.pyplot as plt
import shutil
from cv_bridge import CvBridge

class MULSAInference(pl.LightningModule):
    def __init__(self):
        rospy.init_node('model')
        super().__init__()
        self.transform_image = A.Compose([           
                A.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225], max_pixel_value= 1.0),
                ToTensorV2(),
            ]
        )
        self.root = '/home/soundsense/data/mulsa/dagger'
        self.run_id = 1
        shutil.rmtree(self.root)  # remove dir and all contains
        os.makedirs(self.root + f'/{self.run_id}/video', exist_ok=True)
        os.makedirs(self.root + f'/{self.run_id}/audio', exist_ok=True)
        self.file = open(self.root + f'/{self.run_id}/keyboard_teleop.txt', 'w')
        self.model_name = ''
        self.model_loaded = False
        self.bridge = CvBridge()
    def load_model(self, data):
        if self.model_name == data.data:
            return
        

        print("Received model load request from ", data.data)
        data = data.data
        model_root = "/home/soundsense/models/baselines/mulsa/lightning_logs/"
        dirp, model_name= data.split('/')
        model_root += dirp + '/'#contains ckpt as well 
        config_path = model_root + "hparams.yaml"

        with open(config_path) as info:
            self.config = yaml.load(info.read(), Loader=yaml.FullLoader) 

        v_encoder = make_vision_encoder(self.config['encoder_dim'])
        a_encoder = make_audio_encoder(self.config['encoder_dim'] * self.config['num_stack'], self.config['norm_audio'])
        self.use_audio = "ag" in self.config["modalities"].split("_")
        self.actor = Actor(v_encoder, a_encoder, self.config).to(self.device)
        self.num_stack = self.config['num_stack']
        self.loss_cce = torch.nn.CrossEntropyLoss(weight= torch.tensor([1]*self.config['action_dim']))

        self.audio_processor = AudioProcessor(self.config)
        
        self.h = self.config['resized_height_v']
        self.w = self.config['resized_width_v']
        self.load_state_dict(
            torch.load( 
                model_root + model_name,
                map_location=self.device,
            )['state_dict']
        )
        self.model_loaded = True
        self.model_name = data
        
        print("Model loaded", model_name)
        print("device", self.device)


    def start_pub_sub(self):
        self.model_sub = rospy.Subscriber("/load_model", String, self.load_model)
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

    def process(self, imagedata, audio):
        if not self.model_loaded:
            print("Model not loaded. sorry")
            return
        
        image = self.bridge.imgmsg_to_cv2(imagedata, 'rgb8')
        image = image.astype(np.float32) / 255.0
        time = str(imagedata.header.stamp)
        if self.use_audio:
            audio = np.frombuffer(audio.audio.data, dtype=np.int16).astype(np.float32)
            audio /= 32768.0
            audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            mel = self.audio_processor.process(audio, 0, audio.size(-1)).to(self.device)
            # print(mel.shape)
            
            # plt.imsave(self.root + f'/{self.run_id}/audio/audio.png', mel.squeeze().numpy())
            # sf.write(self.root + f'/{self.run_id}/audio/audio.wav', audio.squeeze().numpy(), self.config['resample_rate_audio'])
            # np.save(self.root + f'/{self.run_id}/audio/{time}.npy', mel.detach().numpy())
        else:
            mel = None

        annotation = str(imagedata.header.frame_id)
        if annotation == 'q':
            self.model_loaded = False
            self.file.close()
            self.run_id += 1    
            os.makedirs(self.root + f'/{self.run_id}/video', exist_ok=True)
            os.makedirs(self.root + f'/{self.run_id}/audio', exist_ok=True)
            self.file = open(self.root + f'/{self.run_id}/keyboard_teleop.txt', 'w')
            return
        print(time, annotation)
        # self.file.write(time + " " + annotation + "\n")
        # cv2.imwrite(self.root + f'/{self.run_id}/video/' + str(time) + '.png', image)
        # cv2.imwrite('temp.png', (image * 255).astype(np.uint8) )
        

        if self.use_audio:
            temp = mel.clone().cpu().detach().numpy().squeeze()
            temp -= temp.min()
            temp /= temp.max()
            temp *= 255
            temp = temp.astype(np.uint8)
            temp = cv2.resize(temp, image.shape[:2][::-1])
            temp = cv2.applyColorMap(temp, cv2.COLORMAP_VIRIDIS)
            temp = cv2.applyColorMap(temp, cv2.COLORMAP_VIRIDIS)
            stacked = np.vstack([image, temp])
        images = [image[:, i * self.w : (i + 1) * self.w] for i in range(self.num_stack)]
        assert len(images) == self.num_stack
        # stacked = np.hstack(images)
        # for i, img in enumerate(images):
        #     plt.imsave(f'temp{i}.png', img)
        out = self.forward({"video": images, "audio": mel})
        out = out.detach().cpu().numpy().squeeze(0)
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
        # print(video.max(), video.min())

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
    import os
    model = MULSAInference()
    model.eval()    
    model.start_pub_sub()

    rospy.spin()