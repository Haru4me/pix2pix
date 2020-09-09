import torch.nn as nn
import torch

"""
Unet Generator:

encoder
C64-C128-C256-C512-C512-C512-C512-C512

decoder
CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
"""

class EncoderLayer(nn.Module):
	
	def __init__(self,inp,outp,norm=True):
		
		super(EncoderLayer, self).__init__()
		
		net = [nn.Conv2d(inp,outp,4, stride = 2,padding = 1)]

		if norm == True:
			net.append(nn.BatchNorm2d(outp))
				
		net.append(nn.LeakyReLU(0.2))
		self.layer = nn.Sequential(*net)
	
	def forward(self,img):
		return self.layer(img)

	
	
class DecoderLayer(nn.Module):
	
	def __init__(self,inp,outp,drop=False):
		
		super(DecoderLayer, self).__init__()
		
		net = [nn.ConvTranspose2d(inp,outp,4, stride = 2,padding = 1)]
		net.append(nn.BatchNorm2d(outp))
			
		if drop == True:
			net.append(nn.Dropout(0.4))
				
		net.append(nn.ReLU())
		self.layer = nn.Sequential(*net)
	
	def forward(self,img,down):
		return torch.cat((self.layer(img), down), 1)

	

class Generator(nn.Module):
	
	def __init__(self):
		
		super(Generator, self).__init__()
		
		self.enc0 = EncoderLayer(3,64,norm=False)
		self.enc1 = EncoderLayer(64,128)
		self.enc2 = EncoderLayer(128,256)
		self.enc3 = EncoderLayer(256,512)
		self.enc4 = EncoderLayer(512,512)
		self.enc5 = EncoderLayer(512,512)
		self.enc6 = EncoderLayer(512,512)
		self.enc7 = EncoderLayer(512,512,norm=False)
		
		self.dec0 = DecoderLayer(512,512,drop=True)
		self.dec1 = DecoderLayer(1024,512,drop=True)
		self.dec2 = DecoderLayer(1024,512,drop=True)
		self.dec3 = DecoderLayer(1024,512)
		self.dec4 = DecoderLayer(1024,256)
		self.dec5 = DecoderLayer(512,128)
		self.dec6 = DecoderLayer(256,64)
		
		self.out = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv2d(128, 3, 5, padding=2),
			nn.Tanh()
		)

		
	def forward(self, img):
		
		e0 = self.enc0(img)
		e1 = self.enc1(e0)
		e2 = self.enc2(e1)
		e3 = self.enc3(e2)
		e4 = self.enc4(e3)
		e5 = self.enc5(e4)
		e6 = self.enc6(e5)
		e7 = self.enc7(e6)
		
		d0 = self.dec0(e7,e6)
		d1 = self.dec1(d0,e5)
		d2 = self.dec2(d1,e4)
		d3 = self.dec3(d2,e3)
		d4 = self.dec4(d3,e2)
		d5 = self.dec5(d4,e1)
		d6 = self.dec6(d5,e0)
		
		return self.out(d6)




"""
70 × 70 discriminator architecture 
C64-C128-C256-C512
"""

class Discriminator(nn.Module):

	def __init__(self):

		super(Discriminator, self).__init__()

		def Ck(inp, outp): # Convolution-BatchNorm-ReLU
			net = [nn.Conv2d(inp, outp, 4, stride=2, padding=1)]
			net.append(nn.BatchNorm2d(outp))
			net.append(nn.ReLU())

			return net

		self.model = nn.Sequential(
			*Ck(6,64),
			*Ck(64,128),
			*Ck(128,512),
			nn.Conv2d(512, 1, 4, padding=1)
		)
		
	def forward(self,imgA,imgB):
		img = torch.cat((imgA,imgB),dim=1)
		return self.model(img)