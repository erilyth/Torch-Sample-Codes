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
require "qtwidget"
require "os"

--w1 = qtwidget.newwindow(300, 300)
--w2 = qtwidget.newwindow(300, 300)
--w3 = qtwidget.newwindow(300, 300) -- Visualize convnet layers

classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

function get_image(cur_image)
	local cur_r = cur_image[{{1,1024}}]
	local cur_g = cur_image[{{1025,2048}}]
	local cur_b = cur_image[{{2049,3072}}]
	-- Generate an image of size 3*32*32
	-- First is the color channel, second is the height and last is the width
	local image = torch.Tensor(3,32,32)
	for j=1,32 do
		for k=1,32 do
			image[1][j][k] = cur_r[(j-1)*32+k]
			image[2][j][k] = cur_g[(j-1)*32+k]
			image[3][j][k] = cur_b[(j-1)*32+k]
		end
	end
	return image
end

function error(a,b)
	local err = 0.0
	for j=1,32 do
		for k=1,32 do
			err = err + math.abs(a[1][j][k] - b[1][j][k])
			err = err + math.abs(a[2][j][k] - b[2][j][k])
			err = err + math.abs(a[3][j][k] - b[3][j][k])
		end
	end
	return err
end

new_model = 1
use_cuda = 0

-- Parametrs for SpatialConvolution = (inputlayers, outputlayers, kernel_width, kernel_height, x_stride, y_stride, x_padding, y_padding)
-- Parametrs for SpatialMaxPooling = (width, height, x_stride, y_stride, x_padding, y_padding)
-- Input = 3*32*32 image, Output = 10 sized vector
if new_model==1 then
    cnn = nn.Sequential()
    cnn:add(nn.View(3,32,32))
    cnn:add(nn.Reshape(3*32*32))
    cnn:add(nn.Linear(3*32*32,3*32*32))
    cnn:add(nn.Reshape(3,32,32))
    cnn:add(nn.SpatialConvolution(3, 16, 3, 3, 1, 1, 1, 1))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
    cnn:add(nn.SpatialConvolution(16, 8, 3, 3, 1, 1, 1, 1))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
    cnn:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    cnn:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialUpSamplingNearest(2))
    cnn:add(nn.SpatialConvolution(8, 8, 3, 3, 1, 1, 1, 1))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialUpSamplingNearest(2))
    cnn:add(nn.SpatialConvolution(8, 16, 3, 3, 1, 1, 1, 1))
    cnn:add(nn.ReLU(true))
    cnn:add(nn.SpatialUpSamplingNearest(2))
    cnn:add(nn.SpatialConvolution(16, 3, 3, 3, 1, 1, 1, 1))
    cnn:add(nn.Sigmoid(true))
    cnn:add(nn.View(3*32*32))
    print('Creating a new network')
else
    cnn = torch.load('model_full.torch')
    print('Using existing network')
end

criterion = nn.MSECriterion()
if use_cuda == 1 then
	criterion = criterion:cuda()
	cnn = cnn:cuda()
end

-- Run the training 'iterations' number of times
iterations = 3

for tt=1,iterations do
	for iti=0,4 do
		data_subset = torch.load('data_batch_' .. (iti+1) .. '.t7', 'ascii')
		image_data = data_subset.data:t()
		image_labels = data_subset.labels[1]
		no_of_training_cases = 10000
	    -- local cur_image = torch.rand(3,32,32)
	    -- for j=1,32 do
		-- 	for k=1,32 do
		--		cur_image[1][j][k] = 1.0
		--		cur_image[2][j][k] = 1.0
		--		cur_image[3][j][k] = 1.0
		--	end
		-- end
	    -- print(cur_image)
        -- Input can be either 3x32x32 or 3*32*32 vectors
	    for p=1,no_of_training_cases do
	        input = image_data[p]:double() / 255.0
	        output = image_data[p]:double() / 255.0
	        -- image.display{image=(image.scale(get_image(output), 300, 300, 'bilinear')), win=w1}
	   	    if use_cuda == 1 then
	   	    	input = input:cuda()
	   	    	output = output:cuda()
	   	    end
	        outputs_cur = cnn:forward(input)
	        local errs = criterion:forward(outputs_cur, output)
	        local df_errs = criterion:backward(outputs_cur, output)
	        -- cnn:zeroGradParameters()
	        inputgrad = cnn:backward(input, df_errs)
	        -- Accumulate gradients and back propogate
	        cnn:updateParameters(0.05)
	        -- input = input - 1 * inputgrad
	        print(tt,iti,p,error(get_image(output),get_image(outputs_cur)))
	        --if p%20 == 1 then
	        --	image.display{image=(image.scale(get_image(input * 255.0), 300, 300, 'bilinear')), win=w2}
        	--	image.display{image=(image.scale(get_image(inputgrad * 255.0), 300, 300, 'bilinear')), win=w3}
        	--end
        end
	end
	print('Done training, saving model if needed')
	torch.save('model_full.torch', cnn)
end

print('Done training, saving model if needed')
torch.save('model_full.torch', cnn)