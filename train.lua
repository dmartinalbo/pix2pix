-- usage example: DATA_ROOT=/path/to/data/ which_direction=BtoA name=expt1 th train.lua 
--
-- code derived from https://github.com/soumith/dcgan.torch
--

require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
require 'image'
require 'models'

opt = {
  DATA_ROOT = '',         -- path to images (should have subfolders 'train', 'val', etc)
  batchSize = 16,          -- # images in batch
  loadSize = 286,         -- scale images to this size
  fineSize = 256,         --  then crop to this size
  ngf = 64,               -- #  of gen filters in first conv layer
  ndf = 64,               -- #  of discrim filters in first conv layer
  input_nc = 1,           -- #  of input image channels
  niter = 200,            -- #  of iter at starting learning rate
  lr = 0.0002,            -- initial learning rate for adam
  beta1 = 0.5,            -- momentum term of adam
  ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
  flip = 1,               -- if flip the images for data argumentation
  display = 1,            -- display samples while training. 0 = false
  display_id = 10,        -- display window id.
  gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
  name = '',              -- name of the experiment, should generally be passed on the command line
  which_direction = 'AtoB',    -- AtoB or BtoA
  phase = 'train',             -- train, val, test, etc
  preprocess = 'regular',      -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
  nThreads = 2,                -- # threads for loading data
  save_epoch_freq = 10,        -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
  display_freq = 10,          -- display the current results every display_freq iterations
  continue_train=0,            -- if continue training, load the latest model: 1: true, 0: false
  serial_batches = 0,          -- if 1, takes images in order to make batches, otherwise takes them randomly
  serial_batch_iter = 1,       -- iter into serial image list
  checkpoints_dir = './checkpoints', -- models are saved here
  cudnn = 1,                         -- set to 0 to not use cudnn (untested)
  condition_GAN = 1,                 -- set to 0 to use unconditional discriminator
  use_GAN = 1,                       -- set to 0 to turn off GAN term
  use_L1 = 1,                        -- set to 0 to turn off L1 term
  which_model_netD = 'basic', -- selects model to use for netD
  which_model_netG = 'unet',  -- selects model to use for netG
  n_layers_D = 0,             -- only used if which_model_netD=='n_layers'
  lambda = 100,               -- weight on L1 term in objective
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
--print(opt)

local input_nc = opt.input_nc
local output_nc = opt.input_nc
-- translation direction
local idx_A = nil
local idx_B = nil

if opt.which_direction=='AtoB' then
  idx_A = {1, input_nc}
  idx_B = {input_nc+1, input_nc+output_nc}
elseif opt.which_direction=='BtoA' then
  idx_A = {input_nc+1, input_nc+output_nc}
  idx_B = {1, input_nc}
else
  error(string.format('bad direction %s',opt.which_direction))
end

if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
--print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local data_loader = paths.dofile('data/data.lua')

-- TODO fix this shit
local data_train = data_loader.new(opt.nThreads, opt)
opt.phase = 'val'
local data_valid = data_loader.new(opt.nThreads, opt)
opt.phase = 'train'

----------------------------------------------------------------------------
local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') then
    m.weight:normal(0.0, 0.02)
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
  end
end

local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

function defineG(input_nc, output_nc, ngf, nz)
  if opt.which_model_netG == "encoder_decoder" then 
    netG = defineG_encoder_decoder(input_nc, output_nc, ngf, nz, 3)
  elseif opt.which_model_netG == "unet" then 
    netG = defineG_unet(input_nc, output_nc, ngf)
  else 
    error("unsupported netG model")
  end

  netG:apply(weights_init)

  return netG
end

function defineD(input_nc, output_nc, ndf)
    
  local netD = nil
  if opt.condition_GAN==1 then
      input_nc_tmp = input_nc
  else
      input_nc_tmp = 0 -- only penalizes structure in output channels
  end
  
  if opt.which_model_netD == "basic" then 
    netD = defineD_basic(input_nc_tmp, output_nc, ndf)
  elseif opt.which_model_netD == "n_layers" then 
    netD = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D)
  else 
    error("unsupported netD model")
  end
  
  netD:apply(weights_init)
  
  return netD
end

-- load saved models and finetune
if opt.continue_train == 1 then
  --print('loading previously trained netG...')
  netG = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), opt)
  --print('loading previously trained netD...')
  netD = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), opt)
else
  --print('define model netG...')
  netG = defineG(input_nc, output_nc, ngf, nz)
  --print('define model netD...')
  netD = defineD(input_nc, output_nc, ndf)
end

--print(netG)
--print(netD)

local criterion = nn.BCECriterion()
local criterionAE = nn.AbsCriterion()
---------------------------------------------------------------------------
optimStateG = {
  learningRate = opt.lr,
  beta1 = opt.beta1,
}
optimStateD = {
  learningRate = opt.lr,
  beta1 = opt.beta1,
}
----------------------------------------------------------------------------
-- local real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
-- local real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
-- local fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
-- local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
-- local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local errD, errG, errL1 = 0, 0, 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------

if opt.gpu > 0 then
  --print('transferring to gpu...')
  require 'cunn'
  cutorch.setDevice(opt.gpu)
  if opt.cudnn==1 then
    netG = util.cudnn(netG); netD = util.cudnn(netD);
  end
  netD:cuda()
  netG:cuda()
  criterion:cuda()
  criterionAE:cuda()
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

function createRealFake(data)
  local real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
  local real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
  local fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
  local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
  local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)

  if opt.gpu > 0 then
    real_A = real_A:cuda()
    real_B = real_B:cuda() 
    fake_B = fake_B:cuda()
    real_AB = real_AB:cuda() 
    fake_AB = fake_AB:cuda()
  end

  -- load real
  data_tm:reset(); data_tm:resume()
  --local real_data, data_path = data:getBatch()
  data_tm:stop()
  
  real_A:copy(data[{ {}, idx_A, {}, {} }])
  real_B:copy(data[{ {}, idx_B, {}, {} }])
  
  if opt.condition_GAN==1 then
      real_AB = torch.cat(real_A,real_B,2)
  else
      real_AB = real_B -- unconditional GAN, only penalizes structure in B
  end
  
  -- create fake
  fake_B = netG:forward(real_A)
  
  if opt.condition_GAN==1 then
      fake_AB = torch.cat(real_A,fake_B,2)
  else
      fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
  end
  -- local predict_real = netD:forward(real_AB)
  -- local predict_fake = netD:forward(fake_AB)
  return real_A, real_B, fake_B, real_AB, fake_AB
end

function train_epoch()
  --------------------------------
  --    TRAINING EPOCH START    --
  --------------------------------
  local train_loss_l1_epoch = 0.0
  epoch_tm:reset()
  local num_batch = 0
  local train_size = math.min(data_train:size(), opt.ntrain)
  for i = 1, train_size, opt.batchSize do
    -- load a batch
    local data_batch, _ = data_train:getBatch()
    local real_A, real_B, fake_B, real_AB, fake_AB = createRealFake(data_batch)

    -- create closure to evaluate f(X) and df/dX of discriminator
    local fDx = function(x)
      netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
      netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
      
      gradParametersD:zero()
      
      -- Real
      local output = netD:forward(real_AB)
      local label = torch.FloatTensor(output:size()):fill(real_label):cuda()
      local errD_real = criterion:forward(output, label)
      local df_do = criterion:backward(output, label)
      netD:backward(real_AB, df_do)
      
      -- Fake
      local output = netD:forward(fake_AB)
      label:fill(fake_label)
      local errD_fake = criterion:forward(output, label)
      local df_do = criterion:backward(output, label)
      netD:backward(fake_AB, df_do)
      
      errD = (errD_real + errD_fake)/2
      
      return errD, gradParametersD
    end

    -- create closure to evaluate f(X) and df/dX of generator
    local fGx = function(x)
      netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
      netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
      
      gradParametersG:zero()
      
      -- GAN loss
      local df_dg = torch.zeros(fake_B:size()):cuda()
      if opt.use_GAN==1 then
        local output = netD.output -- netD:forward{input_A,input_B} was already executed in fDx, so save computation
        local label = torch.FloatTensor(output:size()):fill(real_label):cuda() -- fake labels are real for generator cost
        errG = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        df_dg = netD:updateGradInput(fake_AB, df_do):narrow(2,fake_AB:size(2)-output_nc+1, output_nc)
      else
        errG = 0
      end
      
      -- unary loss
      local df_do_AE = torch.zeros(fake_B:size()):cuda()
      if opt.use_L1==1 then
        errL1 = criterionAE:forward(fake_B, real_B)
        df_do_AE = criterionAE:backward(fake_B, real_B)
      else
        errL1 = 0
      end
      
      netG:backward(real_A, df_dg + df_do_AE:mul(opt.lambda))
      
      return errG, gradParametersG
    end

    -- (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
    if opt.use_GAN==1 then optim.adam(fDx, parametersD, optimStateD) end
    
    -- (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
    optim.adam(fGx, parametersG, optimStateG)
    
    -- Compute epoch (not batch) errors
    train_loss_l1_epoch = train_loss_l1_epoch + errL1
    xlua.progress(i, train_size, opt.ntrain)

    -- display
    if num_batch % opt.display_freq == 0 and opt.display then
      disp.image(
        util.deprocess_batch(util.scaleBatch(real_A:float(),100,100)), 
        {win=opt.display_id, title=opt.name .. ' input'}
        )
      disp.image(
        util.deprocess_batch(util.scaleBatch(fake_B:float(),100,100)), 
        {win=opt.display_id+1, title=opt.name .. ' output'}
        )
      disp.image(
        util.deprocess_batch(util.scaleBatch(real_B:float(),100,100)), 
        {win=opt.display_id+2, title=opt.name .. ' target'}
        )
    end
    num_batch = num_batch + 1
  end
  return train_loss_l1_epoch / train_size, epoch_tm:time().real
end

function valid_epoch()
  --------------------------------
  --   VALIDATION EPOCH START   --
  -------------------------------- 
  local valid_loss_l1_epoch = 0.0
  epoch_tm:reset()
  for i = 1, data_valid:size(), opt.batchSize do
    -- load a batch
    local data_batch, _ = data_valid:getBatch()
    local real_A, real_B, fake_B, real_AB, fake_AB = createRealFake(data_batch)

    local err_L1 = criterionAE:forward(fake_B, real_B)
    -- Compute EPOCH (not batch) errors
    valid_loss_l1_epoch = valid_loss_l1_epoch + err_L1
    xlua.progress(i, data_valid:size())
  end
  return valid_loss_l1_epoch / data_valid:size(), epoch_tm:time().real
end

-- train
paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.name)

for epoch = 1, opt.niter do

  train_loss_l1_epoch, train_duration_epoch = train_epoch()
  valid_loss_l1_epoch, valid_duration_epoch = valid_epoch()

  parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
  parametersG, gradParametersG = nil, nil

  -- Saving stuff  
  if epoch % opt.save_epoch_freq == 0 then
      torch.save(paths.concat(opt.checkpoints_dir, opt.name,  epoch .. '_net_G.t7'), netG:clearState())
      torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D.t7'), netD:clearState())
  end
  
  print(('%d %.3f %.3f %.3f %.3f'):format(
          epoch,
          train_loss_l1_epoch, 
          train_duration_epoch,
          valid_loss_l1_epoch, 
          valid_duration_epoch
          )
  )
  parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
  parametersG, gradParametersG = netG:getParameters()
end