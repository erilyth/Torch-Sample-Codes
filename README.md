# Torch Sample Codes

#### Usage
To run any of the sample codes use:
`th file.lua`

###### Note
To convert CUDA models to non-CUDA, use: `net = cudnn.convert(net, cudnn)`
To convert from normal model to CUDA, use: `netcuda = net:cuda()`

#### Dependencies:
* Torch - luarocks
(Installation Guide - http://torch.ch/docs/getting-started.html)
