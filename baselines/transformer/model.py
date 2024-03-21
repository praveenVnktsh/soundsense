import torch
from vit_pytorch import SimpleViT
import torch.nn.functional as F
import torch.utils.data
## create a train code for images
def compute_loss(xyz_gt, xyz_pred):
    loss = F.mse_loss(xyz_gt, xyz_pred)
    return loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
vit = SimpleViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 5,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
).to(device)
batch = 8
epochs = 10
optimizer = torch.optim.Adam(vit.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=3, gamma=0.9
)
train = torch.randn(800, 3, 256, 256).to(device), torch.randn(800, 5).to(device)
val = torch.randn(240, 3, 256, 256).to(device), torch.randn(240, 5).to(device)

# print(train[0].size(0), val[0].shape)
inputs, xyzgt_gt = train[0], train[1]
val_inputs, val_xyzgt_gt = val[0], val[1]
for k in range(epochs):
    
    for i in range(0, train[0].size(0), batch):
        # print(inputs.shape, xyzgt_gt.shape)
        xyzgt_pred = vit(inputs[i:i+batch])
        loss = compute_loss(xyzgt_gt[i:i+batch], xyzgt_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if k % 10 == 0:
          print(f"epoch {k}, loss {loss.item()}")
    scheduler.step()
    for j in range(0, val[0].size(0), batch):
        xyzgt_pred = vit(val_inputs[j:j+batch])
        loss = compute_loss(val_xyzgt_gt[j:j+batch], xyzgt_pred)
        print(f"val loss {loss.item()}")

      
# save weights after training
torch.save(vit.state_dict(), "vit_weights.pth")


