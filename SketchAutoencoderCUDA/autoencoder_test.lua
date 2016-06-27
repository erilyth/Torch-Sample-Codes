#! /usr/bin/env lua
--
-- nn.lua
-- Copyright (C) 2016 erilyth <erilyth@vishalapr-Lenovo-G50-70>
--
-- Distributed under terms of the MIT license.
--

require "nn"
require "image"
require "cutorch"
require "cunn"
require "qtwidget"
require "os"

correct_matches = 0
accuracy = 0
total = 0

cnn = torch.load('modelcu.torch')

print(cnn)

w = qtwidget.newwindow(100, 100)
w2 = qtwidget.newwindow(100, 100)

no_of_tests = 100
-- Test the network

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

function invertimage(image)
	return (1.0 - image)
end

folder_list, no_of_folders = scandir('./png/')

for i=1,no_of_folders do
	if no_of_tests < 0 then
		break
	end
	folder = folder_list[i]
	images, images_cnt = scandir('./png/' .. folder .. '/')
	for j=images_cnt-10+1,images_cnt do -- Use last 10 images for testing in each class
		no_of_tests = no_of_tests - 1
		if no_of_tests < 0 then
			break
		end
		image_path = './png/' .. folder .. '/' .. images[j]
		local cur_image = image.load(image_path,1,'byte')
		input = cur_image:double()
		input = image.scale(input, 64, 64)
   	    input = input / 255.0
   	    input = invertimage(input)
   	    input = input:cuda()
   	    image.display{image=(input * 255.0), win=w}
        -- Forward prop in the neural network
        local output = cnn:forward(input)
        image.display{image=(output * 255.0), win=w2}
        os.execute("sleep " .. tonumber(0.3))
        print(i,j)
    end
end