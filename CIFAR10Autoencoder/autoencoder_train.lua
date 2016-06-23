#! /usr/bin/env lua
--
-- nn.lua
-- Copyright (C) 2016 erilyth <erilyth@vishalapr-Lenovo-G50-70>
--
-- Distributed under terms of the MIT license.
--

require "nn"
require "image"
require "math"

classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

new_model = 1
torch.setnumthreads(2)

-- Parametrs for SpatialConvolution = (inputlayers, outputlayers, kernel_width, kernel_height, x_stride, y_stride, x_padding, y_padding)
-- Parametrs for SpatialMaxPooling = (width, height, x_stride, y_stride, x_padding, y_padding)
-- Input = 3*32*32 image, Output = 10 sized vector
if new_model==1 then
    cnn = nn.Sequential()
    cnn:add(nn.Reshape(3*32*32))
    cnn:add(nn.Linear(3*32*32,32*32))
    cnn:add(nn.Tanh())
    cnn:add(nn.Linear(32*32,32))
    cnn:add(nn.Tanh())
    cnn:add(nn.Linear(32,32*32))
    cnn:add(nn.Tanh())
    cnn:add(nn.Linear(32*32,3*32*32))
else
    cnn = torch.load('model.torch')
    print('Using existing network')
end

criterion = nn.MSECriterion()

-- Run the training 'iterations' number of times
iterations = 5

for tt=1,iterations do
    print('Epoch = ' .. tt)
	for iti=0,4 do
		data_subset = torch.load('data_batch_' .. (iti+1) .. '.t7', 'ascii')
		image_data = data_subset.data:t()
		image_labels = data_subset.labels[1]
		no_of_training_cases = 10000
	    for p=1,no_of_training_cases do
		    local cur_image = image_data[p]
	        input = cur_image:double()
	   	    input = input / 255.0
	   	    output = input
	        -- Forward prop in the neural network
	        local outputs_cur = cnn:forward(input)
	        local errs = criterion:forward(outputs_cur, output)
	        local df_errs = criterion:backward(outputs_cur, output)
	        -- Reset gradient accumulation
	        cnn:zeroGradParameters()
	        -- Accumulate gradients and back propogate
	        cnn:backward(input, df_errs)
	        -- Update with a learning rate
	        cnn:updateParameters(0.04)
	        print(p)
        end
	end
	print('Save model after each training epoch')
	torch.save('model.torch', cnn)
end

print('Done training, saving model if needed')
torch.save('model.torch', cnn)
