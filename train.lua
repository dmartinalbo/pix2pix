-- usage example: DATA_ROOT=/path/to/data/ which_direction=BtoA name=expt1 th train.lua 
--
-- code derived from https://github.com/soumith/dcgan.torch
--
--require 'laia'
require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
require 'image'
require 'models'

local log = require 'log'
local argparse = require 'argparse'
local opts = require 'data.TrainOptions'

local opt = opts.parse(arg)

--opt.which_model_modelD = 'basic'
--opt.which_model_modelG = 'unet'
opt.preprocess = 'regular'
opt.continue_train = false
opt.condition_GAN = true
opt.use_GAN = true
opt.use_L1 = true
opt.n_layers_D = 0
opt.lambda = 100
opt.serial_batches = 0
opt.serial_batch_iter = 1
opt.ndf = 64
opt.ngf = 64
opt.display_freq = 10
opt.display_id = 10
opt.save_epoch_freq = 10

-- First of all, update logging options and set random seeds
log.loglevel = opt.log_level
log.logfile  = opt.log_file
log.logstderrthreshold = opt.log_stderr_threshold
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

-- Second, set cuda device
if opt.gpu >= 0 then
  cutorch = require 'cutorch'
  cutorch.setDevice(opt.gpu + 1) -- +1 because lua is 1-indexed
end

local input_nc = opt.input_channels
local output_nc = opt.input_channels
-- translation direction
local idx_A = nil
local idx_B = nil

if opt.which_direction=='A2B' then
  idx_A = {1, input_nc}
  idx_B = {input_nc+1, input_nc+output_nc}
elseif opt.which_direction=='B2A' then
  idx_A = {input_nc+1, input_nc+output_nc}
  idx_B = {1, input_nc}
else
  log.error('Bad direction %s', opt.which_direction)
end

-- Determine the filename of the output model
--local output_model_filename = opt.model
if opt.output_model ~= '' then
  local output_model_filename = opt.output_model
end

-- create data loader
local data_loader = paths.dofile('data/data.lua')

local data_train = data_loader.new(opt.training, opt)
local data_valid = data_loader.new(opt.validation, opt)

if opt.num_samples_epoch < 1 or opt.num_samples_epoch > data_train:size() then
  train_num_samples = data_train:size()
else
  train_num_samples = opt.num_samples_epoch
end

local ndf = opt.ndf 
local ngf = opt.ngf 
local real_label = 1
local fake_label = 0

-- -- load saved models and finetune
-- if opt.continue_train then
--   --print('loading previously trained modelG...')
--   modelG = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), opt)
--   --print('loading previously trained modelD...')
--   modelD = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), opt)
-- else
  --print('define model modelG...')
modelG = defineG(input_nc, output_nc, ngf)
  --print('define model modelD...')
modelD = defineD(input_nc, output_nc, ndf)
-- end

local criterion = nn.BCECriterion()
local criterionAE = nn.AbsCriterion()
---------------------------------------------------------------------------
optimStateG = {
  learningRate = opt.learning_rate,
  beta1 = opt.beta1,
}
optimStateD = {
  learningRate = opt.learning_rate,
  beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local errD, errG, errL1 = 0, 0, 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------

if opt.gpu >= 0 then
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  if opt.cudnn then
    require 'cudnn'
    modelG = util.cudnn(modelG)
    modelD = util.cudnn(modelD)
  end
  modelG:cuda()
  modelD:cuda()
  criterion:cuda()
  criterionAE:cuda()
end

local parametersD, gradParametersD = modelD:getParameters()
local parametersG, gradParametersG = modelG:getParameters()

log.info(string.format('Number of parameters in D: %d', parametersD:nElement()))
log.info(string.format('Number of parameters in G: %d', parametersG:nElement()))

if opt.display then 
  disp = require 'display' 
end

function createRealFake(data)
  local real_A = torch.Tensor(opt.batch_size, opt.input_channels, opt.fine_size, opt.fine_size)
  local real_B = torch.Tensor(opt.batch_size, opt.input_channels, opt.fine_size, opt.fine_size)
  local fake_B = torch.Tensor(opt.batch_size, opt.input_channels, opt.fine_size, opt.fine_size)
  local real_AB = nil
  local fake_AB = nil

  if opt.condition_GAN then
    real_AB = torch.Tensor(opt.batch_size, 2 * opt.input_channels, opt.fine_size, opt.fine_size)
    fake_AB = torch.Tensor(opt.batch_size, 2 * opt.input_channels, opt.fine_size, opt.fine_size)
  else
    real_AB = torch.Tensor(opt.batch_size, opt.input_channels, opt.fine_size, opt.fine_size)
    fake_AB = torch.Tensor(opt.batch_size, opt.input_channels, opt.fine_size, opt.fine_size)
  end

  if opt.gpu >= 0 then
    real_A = real_A:cuda()
    real_B = real_B:cuda() 
    fake_B = fake_B:cuda()
    real_AB = real_AB:cuda() 
    fake_AB = fake_AB:cuda()
  end

  -- load real
  data_tm:reset()
  data_tm:resume()
  data_tm:stop()
  
  real_A:copy(data[{ {}, idx_A, {}, {} }])
  real_B:copy(data[{ {}, idx_B, {}, {} }])
  
  if opt.condition_GAN then
      real_AB = torch.cat(real_A,real_B,2)
  else
      real_AB = real_B -- unconditional GAN, only penalizes structure in B
  end
  
  -- create fake
  fake_B = modelG:forward(real_A)
  
  if opt.condition_GAN then
      fake_AB = torch.cat(real_A,fake_B,2)
  else
      fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
  end
  return real_A, real_B, fake_B, real_AB, fake_AB
end

function train_epoch()
  --------------------------------
  --    TRAINING EPOCH START    --
  --------------------------------
  local train_loss_l1_epoch = 0.0
  epoch_tm:reset()
  for batch = 1, train_num_samples, opt.batch_size do
    -- load a batch
    local data_batch, _ = data_train:getBatch()
    local real_A, real_B, fake_B, real_AB, fake_AB = createRealFake(data_batch)

    -- create closure to evaluate f(X) and df/dX of discriminator
    local fDx = function(x)
      modelD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
      modelG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
      
      gradParametersD:zero()
      
      -- Real
      local output = modelD:forward(real_AB)
      local label = torch.FloatTensor(output:size()):fill(real_label) 
      if opt.gpu >= 0 then
        label:cuda()
      end
      local errD_real = criterion:forward(output, label)
      local df_do = criterion:backward(output, label)
      modelD:backward(real_AB, df_do)
      
      -- Fake
      local output = modelD:forward(fake_AB)
      label:fill(fake_label)
      local errD_fake = criterion:forward(output, label)
      local df_do = criterion:backward(output, label)
      modelD:backward(fake_AB, df_do)
      
      errD = (errD_real + errD_fake)/2
      
      return errD, gradParametersD
    end

    -- create closure to evaluate f(X) and df/dX of generator
    local fGx = function(x)
      modelD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
      modelG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
      
      gradParametersG:zero()
      
      -- GAN loss
      local df_dg = torch.zeros(fake_B:size())
      if opt.gpu >= 0 then df_dg:cuda() end
      if opt.use_GAN then
        local output = modelD.output -- modelD:forward{input_A,input_B} was already executed in fDx, so save computation
        local label = torch.FloatTensor(output:size()):fill(real_label)
        if opt.gpu >= 0 then label:cuda() end
        errG = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        df_dg = modelD:updateGradInput(fake_AB, df_do):narrow(2,fake_AB:size(2)-output_nc+1, output_nc)
      else
        errG = 0
      end
      
      -- unary loss
      local df_do_AE = torch.zeros(fake_B:size())
      if opt.gpu >= 0 then df_do_AE:cuda() end
      if opt.use_L1 then
        errL1 = criterionAE:forward(fake_B, real_B)
        df_do_AE = criterionAE:backward(fake_B, real_B)
      else
        errL1 = 0
      end
      
      modelG:backward(real_A, df_dg + df_do_AE:mul(opt.lambda))
      
      return errG, gradParametersG
    end

    -- (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
    if opt.use_GAN then optim.adam(fDx, parametersD, optimStateD) end
    
    -- (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
    optim.adam(fGx, parametersG, optimStateG)
    
    -- Compute epoch (not batch) errors
    train_loss_l1_epoch = train_loss_l1_epoch + errL1
    
    if opt.show_epoch_bar then
      xlua.progress(batch + opt.batch_size -1 , train_num_samples)
    end

    -- display
    if batch % opt.display_freq == 0 and opt.display then
      disp.image(
        util.deprocess_batch(util.scaleBatch(real_A:float(),100,100)), 
        {win=opt.display_id, 'Source'}
        )
      disp.image(
        util.deprocess_batch(util.scaleBatch(fake_B:float(),100,100)), 
        {win=opt.display_id+1, 'Output'}
        )
      disp.image(
        util.deprocess_batch(util.scaleBatch(real_B:float(),100,100)), 
        {win=opt.display_id+2, 'Target'}
        )
    end
  end
  return train_loss_l1_epoch / train_num_samples, epoch_tm:time().real
end

function valid_epoch()
  --------------------------------
  --   VALIDATION EPOCH START   --
  ------------ide-------------------- 
  local valid_loss_l1_epoch = 0.0
  epoch_tm:reset()
  for batch = 1, data_valid:size(), opt.batch_size do
    -- load a batch
    local data_batch, _ = data_valid:getBatch()
    local real_A, real_B, fake_B, real_AB, fake_AB = createRealFake(data_batch)

    local err_L1 = criterionAE:forward(fake_B, real_B)
    -- Compute EPOCH (not batch) errors
    valid_loss_l1_epoch = valid_loss_l1_epoch + err_L1
    
    if opt.show_epoch_bar then
      xlua.progress(batch + opt.batch_size-1 , data_valid:size())
    end
  end
  return valid_loss_l1_epoch / data_valid:size(), epoch_tm:time().real
end

local epoch = 0
while opt.max_epochs <= 0 or epoch < opt.max_epochs do
  -- Epoch starts at 0, when the model is created
  epoch = epoch + 1

  local train_loss_l1_epoch, train_duration_epoch = train_epoch()
  local valid_loss_l1_epoch, valid_duration_epoch = valid_epoch()
  
  parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
  parametersG, gradParametersG = nil, nil

  local best_model = false
  -- local curr_crit_value = current_criterion_value[opt.early_stop_criterion]()
  -- if best_criterion_value == nil or curr_crit_value < best_criterion_value then
  --   best_model = true
  --   if best_criterion_value == nil or
  --   ((best_criterion_value - curr_crit_value) / best_criterion_value) >= opt.min_relative_improv then
  --     last_signif_improv_epoch = epoch
  --   end
  -- end

  -- Saving stuff  
  if best_model then
    best_criterion_value = curr_crit_value
    best_criterion_epoch = epoch
    local checkpoint = {}
    checkpoint.model_opt = initial_checkpoint.model_opt
    checkpoint.train_opt = opt       -- Original training options
    checkpoint.epoch     = epoch
    checkpoint.modelG    = modelG
    checkpoint.modelD    = modelD
    -- Current RNG state
    --checkpoint.rng_state = 
    -- Current rmsprop options (i.e. current learning rate)
    --checkpoint.rmsprop   = rmsprop_opts
    modelG:clearState()
    modelD:clearState()
    -- Only save t7 checkpoint if there is an improvement in L1 loss
    torch.save(output_model_filename, checkpoint)
  end
  
  -- Print progress of the loss function
  if output_progress_file ~= nil then
    output_progress_file:write(
      string.format(
          'Epoch = %d  Loss = %10.4f / %10.4f',
          epoch, 
          train_loss_l1_epoch, 
          valid_loss_l1_epoch
      )
  else
    log.info(
      string.format(
            'Epoch = %d  Loss = %10.4f / %10.4f',
            epoch, 
            train_loss_l1_epoch, 
            valid_loss_l1_epoch
        )
    )
  end

  -- Collect garbage every so often
  if epoch % 5 == 0 then collectgarbage() end

  parametersD, gradParametersD = modelD:getParameters() -- reflatten the params and get them
  parametersG, gradParametersG = modelG:getParameters()
end