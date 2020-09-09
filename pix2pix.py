import numpy as np
import matplotlib.pyplot as plt
import itertools

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


from data import Data
from models import *
import sys


def pic_elustration(epoch, test=False):

	unloader = transforms.ToPILImage()
	
	if test == False:
		dirr = "train_imgs"
	else:
		dirr = "test_imgs"
	
	data = next(iter(validator))
	
	dic = {}
	
	G.eval()
	dic["Real A"] = data["A"]
	dic["Real B"] = data["B"]
	dic["G(A)"] = G(data["A"])
	
	plt.figure(figsize = (12,5))
	
	for i,key in enumerate(dic.keys()):
		
		plt.subplot(1,3,i+1)
		plt.imshow(unloader(dic[key][0]))
		plt.title(key)
		plt.axis('off')
		
	plt.savefig("./{0}/img_{1}.png".format(dirr,epoch), bbox_inches="tight")
	plt.close()



def train(train_data, generator, discriminator, opt_g, opt_d, loss, L1_loss, num_epoch=200, k=1):

	print("Star training...")

	patch = (1, 31, 31) #discr output shape

	for epoch in tqdm(range(num_epoch), desc="Epoch"):
		
		g_loss_run = []
		d_loss_run = []

		for batch in train_data:
			
			realA = batch["A"].to(device)
			realB = batch["B"].to(device)
			genB = generator(realA)
			
			valid = torch.Tensor(np.zeros((realA.shape[0], *patch))).to(device)
			fake = torch.Tensor(np.zeros((realA.shape[0], *patch))).to(device)

			"""
				Train Discriminator
			"""
			
			for _ in range(k):
				
				discriminator.train()
				opt_d.zero_grad()
				
				d_loss = 0.5 * (loss(discriminator(realA,realB), valid) + \
					loss(discriminator(realA,genB.detach()), fake))
				
				d_loss.backward()
				opt_d.step()
				d_loss_run.append(d_loss.item())
				
			"""
				Train Generator
			"""

			generator.train()
			opt_g.zero_grad()
			
			g_loss = loss(discriminator(realA,genB),valid) + 20 * L1_loss(realB, genB)
				
			g_loss.backward()
			opt_g.step()
			g_loss_run.append(g_loss.item())
		
		pic_elustration(epoch)
		
		if epoch % 10 == 0:

			print("D loss: {0}\t||\tG loss: {1}".format(np.mean(d_loss_run),np.mean(g_loss_run)))
			torch.save(generator, f'./saved_model/G_{epoch}.pth')
			torch.save(discriminator, f'./saved_model/D_{epoch}.pth')

	print("Finished!")




if __name__ == "__main__":

	run = sys.argv[1]

	validator = DataLoader(
		Data("./facades/test", swap=True),
		batch_size=1,
		shuffle=True
	)



	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if run == "train":

		dataloader = DataLoader(
			Data("./facades/train", swap=True),
			batch_size=1,
			shuffle=True
		)

		G = Generator().to(device)
		D = Discriminator().to(device)

		loss = nn.MSELoss()
		L1_loss = nn.L1Loss()

		optimizer_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
		optimizer_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
		train(dataloader, G, D, optimizer_G, optimizer_D, loss, L1_loss)

	elif run == "test":

		print("Test starts!")
		path = './saved_model/G_200.pth'
		G = torch.load(path,map_location=device)
		[pic_elustration(i, test=True) for i in tqdm(range(106))]
		print("Finished!")

	else:

		print("Error command!")






