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

i = 0
data_subset = torch.load('data_batch_' .. (i+1) .. '.t7', 'ascii')
image_data = data_subset.data:t()
image_labels = data_subset.labels

idx = 1
cur_image = image_data[idx]
cur_r = cur_image[{{1,1024}}]
cur_g = cur_image[{{1025,2048}}]
cur_b = cur_image[{{2049,3072}}]

-- Generate an image of size 3*32*32
-- First is the color channel, second is the height and last is the width
fin_image = torch.Tensor(3,32,32)
for j=1,32 do
	for k=1,32 do
		fin_image[1][j][k] = cur_r[(j-1)*32+k]
		fin_image[2][j][k] = cur_g[(j-1)*32+k]
		fin_image[3][j][k] = cur_b[(j-1)*32+k]
	end
end

print(fin_image)