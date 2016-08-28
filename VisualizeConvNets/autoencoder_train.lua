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

w1 = qtwidget.newwindow(300, 300)
w2 = qtwidget.newwindow(300, 300)
w3 = qtwidget.newwindow(300, 300) -- Visualize convnet layers

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

new_model = 0
use_cuda = 0

-- Parametrs for SpatialConvolution = (inputlayers, outputlayers, kernel_width, kernel_height, x_stride, y_stride, x_padding, y_padding)
-- Parametrs for SpatialMaxPooling = (width, height, x_stride, y_stride, x_padding, y_padding)
-- Input = 3*32*32 image, Output = 10 sized vector
if new_model==1 then
    cnn = nn.Sequential()
    cnn:add(nn.View(3,32,32))
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
    
else
    cnn = torch.load('model.torch')
    print('Using existing network')
end

criterion = nn.MSECriterion()
if use_cuda == 1 then
	criterion = criterion:cuda()
	cnn = cnn:cuda()
end

-- Run the training 'iterations' number of times
iterations = 1

for tt=1,iterations do
	for iti=0,0 do
		data_subset = torch.load('data_batch_' .. (iti+1) .. '.t7', 'ascii')
		image_data = data_subset.data:t()
		image_labels = data_subset.labels[1]
		no_of_training_cases = 10000
	    local cur_image = torch.rand(3,32,32)
	    -- print(cur_image)
        -- Input can be either 3x32x32 or 3*32*32 vectors
        input = cur_image:double()
        image.display{image=(image.scale(get_image(image_data[5]), 300, 300, 'bilinear')), win=w1}
	    for p=1,no_of_training_cases do
		    -- local cur_image = image_data[p]
	   	    output = image_data[5]:double() / 255.0
	   	    if use_cuda == 1 then
	   	    	input = input:cuda()
	   	    	output = output:cuda()
	   	    end

	   	    cur_input = input
	   	    for layer_idx=1,cnn:size() do
		        -- Forward prop in the neural network
		        outputs_cur = cnn:get(layer_idx):forward(cur_input)
		        cur_input = outputs_cur
		    end

		    cur_output = output
		    for layer_idx=cnn:size(),2,-1 do
		    	local layer = cnn:get(layer_idx)
		    	local prevlayer = cnn:get(layer_idx-1)
		        -- local errs = criterion:forward(outputs_cur, output)
		        local df_errs = criterion:backward(layer.output, cur_output)
		        cur_output = layer:backward(prevlayer.output, df_errs)
		        -- Reset gradient accumulation
		        -- Accumulate gradients and back propogate
		        -- print(layer:backward(input, df_errs))
	    	end
	    	-- This is a 3x32x32 image since its after the View transform
	    	df_errs_final = criterion:backward(cnn:get(1).output, cur_output)
	    	final_img_err = cnn:get(1):backward(input, df_errs_final)
	    	print(final_img_err:sum(), p, "Error: ", error(get_image(output),input))
	    	-- print(df_errs_final:size())
	    	input = input - 0.05*final_img_err
	    	-- image.display{image=(get_image(outputs_cur) * 255.0), win=w}
	        -- os.execute("sleep " .. tonumber(5))
	        if p%20 == 0 then
        		image.display{image=(image.scale(input * 255.0, 300, 300, 'bilinear')), win=w2}
        		image.display{image=(image.scale(final_img_err * 255.0, 300, 300, 'bilinear')), win=w3}
        	end
        end
	end
end

-- print('Done training, saving model if needed')
-- torch.save('model.torch', cnn)