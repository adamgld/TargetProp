require 'image'
require 'torch'
require 'nn'
require 'optim'
require 'noise_injection'
require 'mnist'
require 'plot_results'

-- Scale the dataset [0,256] -> [0,1]
train.data:mul(1/256)
test.data:mul(1/256)

-- Build model
-- -----------
n= 16*16 -- Input size
model = nn.Sequential()
n1 = nn.noise_injection()  -- Gaussian noise injection
l1 = nn.Linear(n,300)      -- Encoder
n2 = nn.noise_injection()  -- Gaussian noise injection
l2 = nn.Linear(300,n)      -- Decoder

-- Weight sharing between Encoder and Decoder
l2.weight:set(l1.weight:t())
l2.gradWeight:set(l1.gradWeight:t())

model:add(n1)
model:add(l1)
model:add(n2)
model:add(l2)

criterion = nn.MSECriterion() -- Mean Square Error criterion

N = 3000 -- Number of training samples used
batch_size = 16 -- Minibatch size
N_batch = N / batch_size

x, dl_dx = model:getParameters()

local counter = 0 -- keeping track of the minibatch position
feval = function(x_new)
	-- Setting the minibatch indexes
	local start_idx = counter + batch_size + 1
	local end_idx = math.min(N, (counter + 1) * batch_size + 1)
	if end_idx == N then
		counter = 0
	else
		counter = counter + 1
	end

	if x ~= x_new then
		x:copy(x_new)
	end
	dl_dx:zero() -- reset the gradient accumulator
	local target = train.data[{{start_idx,end_idx},{}}] 
	local inputs = target
	-- evaluate the loss function and its derivative wrt x, for that minibatch
	local loss_x = criterion:forward(model:forward(inputs), target)
	model:backward(inputs, criterion:backward(model.output, target))
	-- return loss(x) and dloss/dx
	return loss_x, dl_dx
end

sgd_params = {
   learningRate = 8,
   learningRateDecay = 0.002,
   weightDecay = 1e-4/5,
   momentum = 0.0
}

for i = 1,25 do 
	current_loss = 0
	for j = 1,N_batch do
		_,fs = optim.sgd(feval,x,sgd_params)
		current_loss = current_loss + fs[1]
	end
	print(i,'. current loss =' .. current_loss / N_batch)
end

print('Test MSE: ',criterion:forward(model:forward(test.data),test.data))
print('Test MSE (without hidden noise): ', criterion:forward(l2:forward(l1:forward(test.data)),test.data))
plot() -- plot a random test image, and random filters

