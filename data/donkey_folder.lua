
--[[
    This data loader is a modified version of the one from dcgan.torch
    (see https://github.com/soumith/dcgan.torch/blob/master/data/donkey_folder.lua).
    Copyright (c) 2016, Deepak Pathak [See LICENSE file for details]
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.
    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
local log = require 'log'
paths.dofile('dataset.lua')
-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
if not paths.dirp(opt.data) then
  log.error('Did not find directory: ' .. opt.data)
end

-- a cache file of the training metadata (if doesnt exist, will be created)
--local cache = "cache"
--local cache_prefix = opt.data:gsub('/', '_')
--os.execute('mkdir -p cache')
--local trainCache = paths.concat(cache, cache_prefix .. '_trainCache.t7')

--------------------------------------------------------------------------------------------
local input_nc = opt.input_channels -- input channels
local output_nc = opt.input_channels

local loadSize   = {input_nc, opt.load_size}
local sampleSize = {input_nc, opt.fine_size}

local preprocessAandB = function(imA, imB)
  imA = image.scale(imA, loadSize[2], loadSize[2])
  imB = image.scale(imB, loadSize[2], loadSize[2])
  if input_nc == 3 then
    local perm = torch.LongTensor{3, 2, 1}
    imA = imA:index(1, perm)--:mul(256.0): brg, rgb
    imA = imA:mul(2):add(-1)
    imB = imB:index(1, perm)
    imB = imB:mul(2):add(-1)
  elseif input_nc == 1 then
    imA = imA:mul(2):add(-1)
    imB = imB:mul(2):add(-1)
  end
--   print(img:size())
  assert(imA:max()<=1,"A: badly scaled inputs")
  assert(imA:min()>=-1,"A: badly scaled inputs")
  assert(imB:max()<=1,"B: badly scaled inputs")
  assert(imB:min()>=-1,"B: badly scaled inputs")
 
  local oW = sampleSize[2]
  local oH = sampleSize[2]
  local iH = imA:size(2)
  local iW = imA:size(3)
  
  if iH~=oH then     
    h1 = math.ceil(torch.uniform(1e-2, iH-oH))
  end
  
  if iW~=oW then
    w1 = math.ceil(torch.uniform(1e-2, iW-oW))
  end
  if iH ~= oH or iW ~= oW then 
    imA = image.crop(imA, w1, h1, w1 + oW, h1 + oH)
    imB = image.crop(imB, w1, h1, w1 + oW, h1 + oH)
  end
  
  if opt.flip and torch.uniform() > 0.5 then 
    imA = image.hflip(imA)
    imB = image.hflip(imB)
  end
  
  return imA, imB
end

local function loadImageChannel(path)
    local input = image.load(path, 3, 'float')
    input = image.scale(input, loadSize[2], loadSize[2])

    local oW = sampleSize[2]
    local oH = sampleSize[2]
    local iH = input:size(2)
    local iW = input:size(3)
    
    if iH~=oH then     
      h1 = math.ceil(torch.uniform(1e-2, iH-oH))
    end
    
    if iW~=oW then
      w1 = math.ceil(torch.uniform(1e-2, iW-oW))
    end
    if iH ~= oH or iW ~= oW then 
      input = image.crop(input, w1, h1, w1 + oW, h1 + oH)
    end
    
    
    if opt.flip and torch.uniform() > 0.5 then 
      input = image.hflip(input)
    end
    
--    print(input:mean(), input:min(), input:max())
    local input_lab = image.rgb2lab(input)
--    print(input_lab:size())
--    os.exit()
    local imA = input_lab[{{1}, {}, {} }]:div(50.0) - 1.0
    local imB = input_lab[{{2,3},{},{}}]:div(110.0)
    local imAB = torch.cat(imA, imB, 1)
    assert(imAB:max()<=1,"A: badly scaled inputs")
    assert(imAB:min()>=-1,"A: badly scaled inputs")
    
    return imAB
end

--local function loadImage

-- local function loadImage(path)
--    local input = image.load(path, 3, 'float')
--    local h = input:size(2)
--    local w = input:size(3)

--    local imA = image.crop(input, 0, 0, w/2, h)
--    local imB = image.crop(input, w/2, 0, w, h)
   
--    return imA, imB
-- end

local function loadImage(path, input_nc)
  assert(input_nc ~= 1 or input_nc ~=3, 'Incorrect input channel number')
  local input = image.load(path, input_nc, 'float')
  local h = input:size(2)
  local w = input:size(3)

  local imA = image.crop(input, 0, 0, w/2, h)
  local imB = image.crop(input, w/2, 0, w, h)

  return imA, imB
end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
  collectgarbage()
  if opt.preprocess == 'regular' then
    local imA, imB = loadImage(path)
    imA, imB = preprocessAandB(imA, imB)
    imAB = torch.cat(imA, imB, 1)
  end

  if opt.preprocess == 'colorization' then 
    imAB = loadImageChannel(path)
  end
  return imAB
end

trainLoader = dataLoader{
  paths = {opt.data},
  loadSize = {input_nc, loadSize[2], loadSize[2]},
  sampleSize = {input_nc+output_nc, sampleSize[2], sampleSize[2]},
  split = 100,
  serial_batches = opt.serial_batches, 
  verbose = true
}

trainLoader.sampleHookTrain = trainHook
collectgarbage()

-- do some sanity checks on trainLoader
do
  local class = trainLoader.imageClass
  local nClasses = #trainLoader.classes
  assert(class:max() <= nClasses, "class logic has error")
  assert(class:min() >= 1, "class logic has error")
end