import torch 
import torch.nn as nn 
from torch.optim.lr_scheduler import StepLR


class BasicBlock(nn.Module):
	'''
	This basic block will  be used throughout the network
	'''
	def __init__(self, in_channels, out_channels, identity_downsample = None, stride = 1):
		super(BasicBlock, self).__init__()
		#self.expansion = 1
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_cahnnels, out_channels, kernel_size = 3, stride = stride, padding = 1)
		self.bn2 = nn.BatchNorm(out_channels) 
		self.relu = nn.ReLU()
		self.identity_donwsample = identity_downsample #Convolutional identity block


	def forward(self, x):
		identity = x 
		x = self.conv1(x)
		x = self.bn1(x) # We use BN before activation
		x = self.relu(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)


		if self.identity_downsample is not None: #if this is true, we need to change the dimensions 
			identity = self.identity_downsample(identity)

		x += identity
		x = self.relu(x)
		return x


class ResNet(nn.Module):
	'''
	layers : list(how many times we want to use the block we defined above)
	resnet 20: [3, 3, 3]
	'''

	def __init__(self, basicblock, layers, img_cahnnels, num_classes):
		super(ResNet, self).__init__()
		self.in_channels = 16
		self.conv1 = nn.Conv2d(img_channels, 16, kernel_size = 3, stride = 1, padding = 1)
		self.bn1 = nn.BatchNorm2d(self.in_cahnnels)
		self.relu = nn.ReLU()

		#ResNet layers(18 layers here)
		self.layer1 = self._make_layer(basicblock, num_residual_blocks = layers[0], out_channels = 16) #6 layers
		self.layer2 = self._make_layer(basicblock, num_residual_blocks = layers[1], out_channels = 32, stride = 2)
		self.layer3 = self._make_layer(basicblock, num_residual_blocks = layers[2], out_channels = 64, stride = 2)
		
		self.avg_pool = nn.AvgPool2d((8, 8)) #For resnet 50 or more, it should get AdaptivePool 
		self.fc = nn.Linear(64, num_classes)


	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
	
		x = self.avg_pool(x)
		x = x.reshape(x.shape[0], -1)
		x = self.fc(x)

		return x 


	def _make_layer(self, basicblock, num_residual_blocks, out_channels, stride = 1):
		identity_downsample = None 
		layers = []

		if stride != 1 or self.in_cahnnels != out_channels: # gonna run only two times in Res20
			identity_downsample = nn.Sequential(nn.Conv2d(self.in_cahnnels, out_channels*2, kernel_size = 3, stride = 2), 
												nn.BatchNorm2d(out_channels*2))


		layers.append(basicblock(self.in_cahnnels, out_channels, identity_downsample, stride))
		self.in_channels = out_channels #Updating self in_channels after the number of channels is increased
		
		for i in range(num_residual_blocks - 1): #(gonna be 2--> 3-1=2)
			layers.append(basicblock(out_channels, out_channels))

		return nn.Sequential(*layers)






'''
def ResNet20(img_channels = 3, num_classes = 3):
	return ResNet(BasicBlock, layers = [3, 3, 3], img_cahnnels = img_channels, num_classes = num_classes)
'''
train_loader = torch.utils.data.DataLoader(transformed_cifar10, batch_size = 64, shuffle = True)


model = ResNet(BasicBlock, layers = [3, 3, 3], img_cahnnels = 3, num_classes = 10)



#LOSS AND OPTIMIZER ----------------------------------------------------------
learning_rate = 0.001
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-4)

scheduler = StepLR(optimizer, step_size = 20, gamma = 0.1)





#TRAINING THE MODEL ----------------------------------------------------------
losses = []
accuracy = []

for epoch in range(num_epochs):
	running_loss = 0
	accuracy = 0
	for i, (images, labels) in enumerate(train_loader):
		if torch.cuda.is_available:
			images = images.to(device)
			labels = labels.to(device)


		#FORWARD PASS
		outputs = model(images)
		loss = criterion(outputs, labels)


		#BACKWARD AND OPTIMIZE 
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


		if (i + 1) % 5 == 0:
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
				  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))














