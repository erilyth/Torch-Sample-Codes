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
require "cutorch"
require "cunn"
require "qtwidget"
require "os"

new_model = 1
torch.setnumthreads(2)

w = qtwidget.newwindow(100, 100)
w2 = qtwidget.newwindow(100, 100)
w3 = qtwidget.newwindow(100, 100)

function invertimage(image)
	return (1.0 - image)
end

function stochasticNoise(image)
	gen = torch.Generator()
	new_image = torch.Tensor(1,64,64)
	new_image = image
	torch.manualSeed(gen, 0)
	for j=1,64 do
		for k=1,64 do
			val = torch.uniform()
			val2 = torch.uniform()
			if val >= 0.988 then
				new_image[1][j][k] = val2
			end
		end
	end
	return new_image
end

function scandir(directory)
    local i, t, popen = 0, {}, io.popen
    local command = 'ls -a "' .. directory .. '"'
    local pfile,err = popen(command)
    local no_of_objects = 0
    for filename in pfile:lines() do
    	i = i + 1
    	if i >= 3 then
    		no_of_objects = no_of_objects + 1
       		t[i-2] = filename
       	end
    end
    pfile:close()
    return t, no_of_objects
end

folder_list, no_of_folders = scandir('./png/')

-- Parametrs for SpatialConvolution = (inputlayers, outputlayers, kernel_width, kernel_height, x_stride, y_stride, x_padding, y_padding)
-- Parametrs for SpatialMaxPooling = (width, height, x_stride, y_stride, x_padding, y_padding)
-- Input = 3*32*32 image, Output = 10 sized vector
if new_model==1 then
    cnn = nn.Sequential()
    cnn:add(nn.Reshape(64*64))
    cnn:add(nn.Linear(64*64,2048))
    cnn:add(nn.Tanh())
    cnn:add(nn.Linear(2048,512))
    cnn:add(nn.Tanh())
    cnn:add(nn.Linear(512,2048))
    cnn:add(nn.Tanh())
    cnn:add(nn.Linear(2048,64*64))
    cnn:add(nn.Tanh())
	cnn:add(nn.Reshape(1,64,64))
else
    cnn = torch.load('modelcu.torch')
    print('Using existing network')
end

cnn = cnn:cuda()
criterion = nn.MSECriterion()
criterion = criterion:cuda()

-- Run the training 'iterations' number of times
iterations = 10
updateRate = 1.5
updateEndRate = 1
updateDecay = (updateRate-updateEndRate)/iterations

for tt=1,iterations do
    print('Epoch = ' .. tt)
    updateRate = updateRate - updateDecay
	for i=1,no_of_folders do
		folder = folder_list[i]
		images, images_cnt = scandir('./png/' .. folder .. '/')
		for j=1,images_cnt-10 do -- Use last 10 images for testing in each class
			image_path = './png/' .. folder .. '/' .. images[j]
			local cur_image = image.load(image_path,1,'byte')
			local input = cur_image:double()
			input = image.scale(input, 64, 64)
	   	    input = invertimage(input / 255.0)
	   	    input = stochasticNoise(input)
	   	    input = input:cuda()
	   	    local output = image.load(image_path,1,'byte'):double()
	   	    output = image.scale(output, 64, 64)
	   	    output = invertimage(output / 255.0)
	   	    output = output:cuda()
	   	    -- print(output)
	        -- Forward prop in the neural network
	        local outputs_cur = cnn:forward(input)
	        local errs = criterion:forward(outputs_cur, output)
	        local df_errs = criterion:backward(outputs_cur, output)
	        -- Reset gradient accumulation
	        cnn:zeroGradParameters()
	        -- Accumulate gradients and back propogate
	        cnn:backward(input, df_errs)
	        image.display{image=(input * 255.0), win=w}
	        image.display{image=(output * 255.0), win=w2}
	        image.display{image=(outputs_cur * 255.0), win=w3}
	        -- Update with a learning rate
	        cnn:updateParameters(updateRate)
	        print(i,j,updateRate,tt)
	    end
	end
	print('Save model after each training epoch')
	torch.save('modelcu.torch', cnn)
end

print('Done training, saving model if needed')
torch.save('modelcu.torch', cnn)
