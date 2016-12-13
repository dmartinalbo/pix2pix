--[[
    This data loader is a modified version of the one from dcgan.torch
    (see https://github.com/soumith/dcgan.torch/blob/master/data/data.lua).
    
    Copyright (c) 2016, Deepak Pathak [See LICENSE file for details]
]]--

local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local data = {}

local result = {}
local unpack = unpack and unpack or table.unpack

function data.new(path, options)
  local options = options or {}
  local self = {}
  for k,v in pairs(data) do
    self[k] = v
  end

  local donkey_file = 'donkey_folder.lua'
  local n = options.threads
  options.data = path

  if n > 0 then
    -- local options = opt_
    -- options.data = path_
    self.threads = Threads(n,
      function() require 'torch' end,
      function(idx)
        opt = options
        tid = idx
        local seed = (opt.seed and opt.seed or 0) + idx
        torch.manualSeed(seed)
        torch.setnumthreads(1)
        assert(options, 'Options not found')
        assert(opt, 'Opt not given')
        paths.dofile(donkey_file)
      end
    )
  else
    opt=options
    if donkey_file then paths.dofile(donkey_file) end
    self.threads = {}
    function self.threads:addjob(f1, f2) f2(f1()) end
    function self.threads:dojob() end
    function self.threads:synchronize() end
   end
  
   local nSamples = 0
   self.threads:addjob(function() return trainLoader:size() end,
         function(c) nSamples = c end)
   self.threads:synchronize()
   self._size = nSamples

   for i = 1, n do
      self.threads:addjob(self._getFromThreads,
                          self._pushResult)
   end
--   print(self.threads)
   return self
end

function data._getFromThreads()
   assert(opt.batch_size, 'opt.batch_size not found')
   return trainLoader:sample(opt.batch_size)
end

function data._pushResult(...)
   local res = {...}
   if res == nil then
      self.threads:synchronize()
   end
   result[1] = res
end

function data:getBatch()
  -- queue another job
  self.threads:addjob(self._getFromThreads, self._pushResult)
  self.threads:dojob()
  local res = result[1]

  img_data = res[1]
  img_paths = res[3]

  result[1] = nil
  if torch.type(img_data) == 'table' then
    img_data = unpack(img_data)
  end
  return img_data, img_paths
end

function data:size()
   return self._size
end

return data
