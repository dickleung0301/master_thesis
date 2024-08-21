import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

# state the hyperparameters
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10 

# check the device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# define the autoencoder

class AutoEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid()
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

# load the dataset

dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)

# set up the model, loss function and optimizer

model = AutoEncoder()
model.to(device=device)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training loop 
outputs = []
losses = []

for epoch in range(NUM_EPOCHS):
    for image, _ in dataloader:
        
        # reshape the image into (batch, 28 * 28)
        image = image.reshape(-1, 28 * 28).to(device)

        # output
        reconstruction = model(image)

        # calculating the loss
        loss = criterion(reconstruction, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    outputs.append((epoch, image.to('cpu'), reconstruction.to('cpu')))

# plot the losses
figure, axis = plt.subplots(3, 1)
axis[0].plot(losses)
axis[0].set_title("losses")

# to visualise the reconstruction
image = outputs[-1][1][0]
image = image.reshape(28, 28)
axis[1].imshow(image.numpy())
axis[1].set_title("Original Image")

recon = outputs[-1][2][0]
recon = recon.reshape(28, 28)
axis[2].imshow(recon.detach().numpy())
axis[2].set_title("Reconstructed Image")
plt.show()