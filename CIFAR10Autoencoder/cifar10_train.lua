#! /usr/bin/env lua
--
-- nn.lua
-- Copyright (C) 2016 erilyth <erilyth@vishalapr-Lenovo-G50-70>
--
-- Distributed under terms of the MIT license.
--

require "nn"
require "image"

classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

new_model = 0
torch.setnumthreads(2)

-- Parametrs for SpatialConvolution = (inputlayers, outputlayers, kernel_width, kernel_height, x_stride, y_stride, x_padding, y_padding)
-- Parametrs for SpatialMaxPooling = (width, height, x_stride, y_stride, x_padding, y_padding)
-- Input = 3*32*32 image, Output = 10 sized vector
if new_model==1 then
    cnn = nn.Sequential()
    cnn:add(nn.SpatialConvolution(3,32,5,5,1,1,2,2))
    cnn:add(nn.Tanh())
    cnn:add(nn.SpatialMaxPooling(2,2,2,2))
    cnn:add(nn.SpatialConvolution(32,64,5,5,1,1,2,2))
    cnn:add(nn.Tanh())
    cnn:add(nn.SpatialConvolution(64,128,5,5,1,1,2,2))
    cnn:add(nn.Tanh())
    cnn:add(nn.SpatialMaxPooling(2,2,2,2))
    cnn:add(nn.SpatialConvolution(128,512,3,3,1,1,1,1))
    cnn:add(nn.Tanh())
    cnn:add(nn.Dropout(0.2))
    cnn:add(nn.SpatialConvolution(512,512,3,3,1,1,1,1))
    cnn:add(nn.Tanh())
    cnn:add(nn.Reshape(512*8*8))
    cnn:add(nn.Linear(512*8*8,512))
    cnn:add(nn.Dropout(0.3))
    cnn:add(nn.Tanh())
    cnn:add(nn.Linear(512,10))
    cnn:add(nn.Tanh())
else
    cnn = torch.load('model.torch')
end

criterion = nn.MSECriterion()

-- Run the training 'iterations' number of times
iterations = 1

for tt=1,iterations do
    print('Epoch = ' .. tt)
	for iti=0,4 do
		data_subset = torch.load('data_batch_' .. (iti+1) .. '.t7', 'ascii')
		image_data = data_subset.data:t()
		image_labels = data_subset.labels[1]
		no_of_training_cases = 10000
	    for p=1,no_of_training_cases do
		    local cur_image = image_data[p]
			local cur_r = cur_image[{{1,1024}}]
			local cur_g = cur_image[{{1025,2048}}]
			local cur_b = cur_image[{{2049,3072}}]
			-- Generate an image of size 3*32*32
			-- First is the color channel, second is the height and last is the width
			local input = torch.Tensor(3,32,32)
			for j=1,32 do
				for k=1,32 do
					input[1][j][k] = cur_r[(j-1)*32+k]
					input[2][j][k] = cur_g[(j-1)*32+k]
					input[3][j][k] = cur_b[(j-1)*32+k]
				end
			end
		    local output_val = image_labels[p]
	        local output = torch.Tensor(10);
	        input = input:double()
	   	    input = input / 255.0
	        for k=1,10 do
	            output[k] = 0.0
	        end
	        output[output_val+1] = 1.0
	        -- Forward prop in the neural network
	        local outputs_cur = cnn:forward(input)
	        local errs = criterion:forward(outputs_cur, output)
	        local df_errs = criterion:backward(outputs_cur, output)
	        -- Reset gradient accumulation
	        cnn:zeroGradParameters()
	        -- Accumulate gradients and back propogate
	        cnn:backward(input, df_errs)
	        -- Update with a learning rate
	        cnn:updateParameters(0.01)
	        print(p .. " => " .. output_val)
        end
	end
end

print('Done training, saving model if needed')
torch.save('model.torch', cnn)
