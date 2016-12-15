#!/usr/bin/env th

require 'laia'
require 'optim'
util = paths.dofile('util/util.lua')
require 'models'

local parser = laia.argparse(){
  name = 'train',
  description = ''
}

-- train arguments
parser:argument('checkpoint', 'Input model or checkpoint for training.')
parser:argument('training', 'Location of training images.')
parser:argument('validation', 'Location of validation images.')
parser:argument('which_direction', 'Which translation direction: A2B or B2A.')

-- Custom options
parser:option(
  '--batch_size -b', 'Batch size.',
  9, laia.toint)
parser:option(
  '--load_size', 'Scale images to this size...',
  286, laia.toint)
  :gt(0.0)
parser:option(
  '--fine_size', '...then crop to this size.',
  256, laia.toint)
  :gt(0.0)
parser:option(
  '--flip', 'Flip the images for data argumentation',
  true, toboolean)
  :argname('<bool>')
parser:option(
  '--num_samples_epoch',
  'Number of training samples to process in each epoch; ' ..
    'If n=0, this value is equal to the number of samples in the training '..
    'partition.', 0, laia.toint)
  :argname('<n>')
  :ge(0)
parser:option(
  '--max_epochs', 'If n>0, training will continue for, at most, n epochs.',
  0, laia.toint)
  :argname('<n>')
parser:option(
  '--learning_rate', 'Initial learning rate.', 0.001, tonumber)
  :gt(0.0)
parser:option(
  '--beta1', 'Momentum term of adam', 0.5, tonumber)
  :gt(0.0)
parser:option(
  '--lambda', 'Weight on L1 term in objective', 100, laia.toint)
parser:option(
  '--L1_loss', 'Use L1 term in loss function',
  true, toboolean)
  :argname('<bool>')
parser:option(
  '--GAN', 'Use GAN',
  true, toboolean)
  :argname('<bool>')
parser:option(
  '--conditional_GAN', 'Use conditional GAN',
  true, toboolean)
  :argname('<bool>')
parser:option(
  '--threads', 'Number of threads for data loading', 2, laia.toint)
  :gt(0)
parser:option(
  '--checkpoint_output', 'Save checkpoints to this file. If not given, ' ..
    'the input checkpoint will be overwritten.', '')
  :argname('<file>')
parser:option(
  '--display', 'Display samples while training.',
  true, toboolean)
  :argname('<bool>')
parser:option('--output_progress', 'Write the progress of training ' ..
  'after each epoch to this text file','')
  :argname('<file>')
parser:option(
  '--display_frequency', 'Display current batch every n batches',
  10, laia.toint)
  :argname('<n>')
parser:option(
  '--checkpoint_save_interval', 'Save a checkpoint to disk on every n ' ..
    'epochs. Note: regardless of this, every time a better model is found, ' ..
    'a checkpoint is saved.', 50, laia.toint)
  :argname('<n>')
  :gt(0)
parser:option(
  '--show_epoch_bar',
  'Display a progress bar on the ' ..
    'terminal showing the progress on the train and validation ' ..
    'partitions for each epoch',
  true, toboolean)
  :argname('<bool>')
parser:option(
  '--seed -s', 'Seed for random numbers generation.',
  0x012345, laia.toint)
parser:option(
  '--gpu', 'If gpu>0, uses the specified GPU, otherwise uses the CPU.',
  1, laia.toint)

-- Register logging options
laia.log.registerOptions(parser)

-- Parse from command line
local opt = parser:parse()

-- Initialize random seeds
laia.manualSeed(opt.seed)

opt.cudnn = true
--opt.preprocess = 'regular'
opt.continue_train = false
opt.serial_batches = 0
--opt.serial_batch_iter = 1
opt.save_epoch_freq = 10

-- Load initial checkpoint
laia.log.info('Loading models from %s', opt.checkpoint)
local initial_checkpoint = util.loadCheckpoint(opt)

local modelG = initial_checkpoint.best.modelG
local modelD = initial_checkpoint.best.modelD

local input_nc = initial_checkpoint.model_opt.input_channels
local output_nc = initial_checkpoint.model_opt.input_channels

opt.input_channels = initial_checkpoint.model_opt.input_channels

-- create data loader
local data_loader = paths.dofile('data/data.lua')
local data_train = data_loader.new(opt.training, opt)
local data_valid = data_loader.new(opt.validation, opt)

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
  laia.log.error('Bad direction %s', opt.which_direction)
end

-- Determine the filename of the output model
local output_model_filename = opt.mocdl
if opt.checkpoint_output ~= '' then
  local output_model_filename = opt.checkpoint_output
end

local output_progress_file = nil
if opt.output_progress ~= '' then
  output_progress_file = io.open(opt.output_progress,
         (epoch == 0 and 'w') or 'a')
 -- output_progress_file:write('# EPOCH   BEST?   TRAIN_LOSS   VALID_LOSS   TRAIN_CER   VALID_CER   TRAIN_TIME(min)   VALID_TIME(min)\n')
  output_progress_file:flush()
end

if opt.num_samples_epoch == 0 then
  train_num_samples = data_train:size()
else
  train_num_samples = opt.num_samples_epoch
end

local real_label = 1
local fake_label = 0

local criterion = nn.BCECriterion()
local criterionAE = nn.AbsCriterion()
---------------------------------------------------------------------------
local optimStateG = {
  learningRate = opt.learning_rate,
  beta1 = opt.beta1,
}
local optimStateD = {
  learningRate = opt.learning_rate,
  beta1 = opt.beta1,
}

----------------------------------------------------------------------------
local errD, errG, errL1 = 0, 0, 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------

if opt.gpu > 0 then
  cutorch.setDevice(opt.gpu)
  criterion:cuda()
  criterionAE:cuda()
end

local parametersD, gradParametersD = modelD:getParameters()
local parametersG, gradParametersG = modelG:getParameters()

laia.log.info('Number of parameters in D: %d', parametersD:nElement())
laia.log.info('Number of parameters in G: %d', parametersG:nElement())

if opt.display then 
  disp = require 'display' 
end

function createRealFake(data)
  local real_A = torch.Tensor(opt.batch_size, opt.input_channels, opt.fine_size, opt.fine_size)
  local real_B = torch.Tensor(opt.batch_size, opt.input_channels, opt.fine_size, opt.fine_size)
  local fake_B = torch.Tensor(opt.batch_size, opt.input_channels, opt.fine_size, opt.fine_size)
  local real_AB = nil
  local fake_AB = nil

  if opt.conditional_GAN then
    real_AB = torch.Tensor(opt.batch_size, 2 * opt.input_channels, opt.fine_size, opt.fine_size)
    fake_AB = torch.Tensor(opt.batch_size, 2 * opt.input_channels, opt.fine_size, opt.fine_size)
  else
    real_AB = torch.Tensor(opt.batch_size, opt.input_channels, opt.fine_size, opt.fine_size)
    fake_AB = torch.Tensor(opt.batch_size, opt.input_channels, opt.fine_size, opt.fine_size)
  end

  if opt.gpu > 0 then
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
  
  if opt.conditional_GAN then
      real_AB = torch.cat(real_A,real_B,2)
  else
      real_AB = real_B -- unconditional GAN, only penalizes structure in B
  end
  
  -- create fake
  fake_B = modelG:forward(real_A)
  
  if opt.conditional_GAN then
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
  local train_loss_D_epoch = 0.0
  local train_loss_G_epoch = 0.0
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
      
      if opt.gpu > 0 then 
        label=label:cuda()
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
      if opt.gpu > 0 then df_dg = df_dg:cuda() end
      if opt.GAN then
        local output = modelD.output -- modelD:forward{input_A,input_B} was already executed in fDx, so save computation
        local label = torch.FloatTensor(output:size()):fill(real_label)
        if opt.gpu > 0 then label=label:cuda() end
        errG = criterion:forward(output, label)
        local df_do = criterion:backward(output, label)
        df_dg = modelD:updateGradInput(fake_AB, df_do):narrow(2,fake_AB:size(2)-output_nc+1, output_nc)
      else
        errG = 0
      end
      
      -- unary loss
      local df_do_AE = torch.zeros(fake_B:size())
      if opt.gpu > 0 then df_do_AE = df_do_AE:cuda() end
      if opt.L1_loss then
        -- losses are averaged for each minibatch
        errL1 = criterionAE:forward(fake_B, real_B)
        df_do_AE = criterionAE:backward(fake_B, real_B)
      else
        errL1 = 0
      end
      
      modelG:backward(real_A, df_dg + df_do_AE:mul(opt.lambda))
      
      return errG, gradParametersG
    end

    -- (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
    if opt.GAN then optim.adam(fDx, parametersD, optimStateD) end
    
    -- (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
    optim.adam(fGx, parametersG, optimStateG)
    
    -- Compute epoch (not batch) errors
    train_loss_l1_epoch = train_loss_l1_epoch + errL1 * opt.batch_size
    train_loss_G_epoch  = train_loss_G_epoch + errG * opt.batch_size
    train_loss_D_epoch  = train_loss_D_epoch + errD * opt.batch_size

    if opt.show_epoch_bar then
      xlua.progress(batch + opt.batch_size -1 , train_num_samples)
    end

    -- display
    if batch % opt.display_frequency == 0 and opt.display then
      disp.image(
        util.deprocess_batch(util.scaleBatch(real_A:float(),200,200)), 
        {win=0, 'Source'}
        )
      disp.image(
        util.deprocess_batch(util.scaleBatch(fake_B:float(),200,200)), 
        {win=1, 'Output'}
        )
      disp.image(
        util.deprocess_batch(util.scaleBatch(real_B:float(),200,200)), 
        {win=2, 'Target'}
        )
    end
    if laia.SignalHandler.ExitRequested() then break end
  end
  
  train_loss_l1_epoch = train_loss_l1_epoch / train_num_samples
  train_loss_G_epoch = train_loss_G_epoch / train_num_samples
  train_loss_D_epoch = train_loss_D_epoch / train_num_samples
  
  return train_loss_l1_epoch, train_loss_G_epoch, train_loss_D_epoch, epoch_tm:time().real
end

function valid_epoch()
  --------------------------------
  --   VALIDATION EPOCH START   --
  -------------------------------- 
  local valid_loss_l1_epoch = 0.0
  local valid_loss_G_epoch = 0.0
  local valid_loss_D_epoch = 0.0
  epoch_tm:reset()
  for batch = 1, data_valid:size(), opt.batch_size do
    -- load a batch
    local data_batch, _ = data_valid:getBatch()
    local real_A, real_B, fake_B, real_AB, fake_AB = createRealFake(data_batch)

    local err_L1 = criterionAE:forward(fake_B, real_B)

    -- Real
    local output = modelD:forward(real_AB)
    local label = torch.FloatTensor(output:size()):fill(real_label) 
    
    if opt.gpu > 0 then label=label:cuda() end

    -- calculate losses
    local err_D_real = criterion:forward(output, label)
          
    -- Fake
    output = modelD:forward(fake_AB)
    label:fill(fake_label)
    local err_D_fake = criterion:forward(output, label)

    local err_D = (err_D_real + err_D_fake)/2
    local err_G = criterion:forward(output, label)
     
    -- Compute EPOCH (not batch) errors
    valid_loss_l1_epoch = valid_loss_l1_epoch + err_L1 * opt.batch_size
    valid_loss_G_epoch  = valid_loss_G_epoch + err_G * opt.batch_size
    valid_loss_D_epoch  = valid_loss_D_epoch + err_D * opt.batch_size
    
    if opt.show_epoch_bar then
      xlua.progress(batch + opt.batch_size-1 , data_valid:size())
    end
    if laia.SignalHandler.ExitRequested() then break end
  end
  valid_loss_l1_epoch = valid_loss_l1_epoch / data_valid:size()
  valid_loss_G_epoch = valid_loss_G_epoch / data_valid:size()
  valid_loss_D_epoch = valid_loss_D_epoch / data_valid:size()
  return valid_loss_l1_epoch, valid_loss_G_epoch, valid_loss_D_epoch, epoch_tm:time().real
end

local best = {}

local epoch = 0
local checkpoint = {}
while opt.max_epochs <= 0 or epoch < opt.max_epochs do
  if laia.SignalHandler.ExitRequested() then break end
  -- Epoch starts at 0, when the model is created
  epoch = epoch + 1

  local train_loss_l1_epoch, train_loss_G_epoch, train_loss_D_epoch, train_duration_epoch = train_epoch()
  local valid_loss_l1_epoch, valid_loss_G_epoch, valid_loss_D_epoch, valid_duration_epoch = valid_epoch()
  if laia.SignalHandler.ExitRequested() then break end
  
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
  -- checkpoint = {}
  -- if best_model then
  --   checkpoint.model_opt   = initial_checkpoint.model_opt
  --   checkpoint.train_opt   = opt       -- Original training options
  --   checkpoint.epoch       = epoch
  --   best.modelD = modelD
  --   best.modelG = modelG
  --   checkpoint.best = best
  --   modelD:clearState()
  --   modelG:clearState()
  -- end

  -- -- Save checkpoint: every --checkpoint_save_interval or 
  -- -- when a new better model is found.
  -- if epoch % opts.checkpoint_save_interval == 0 or best_model then
  --   checkpoint.last.modelD = modelD
  --   checkpoint.last.modelG = modelG
  --   modelD:clearState()
  --   modelG:clearState()
  --   checkpoint.best = best
  --   checkpoint.opt = opt
  --   local checkpoint_filename = (opts.output_checkpoint ~= '' and
  --     opts.output_checkpoint) or opts.checkpoint
  --     laia.log.info('Epoch %d, saving checkpoint to %q.',
  --       epoch, checkpoint_filename)
  --     torch.save(checkpoint_filename, checkpoint)
  -- end
  
  -- Print progress of the loss function
  if output_progress_file ~= nil then
    output_progress_file:write(
      string.format(
        '%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f',
        epoch,
        train_loss_G_epoch,
        valid_loss_G_epoch,
        train_loss_D_epoch,
        valid_loss_D_epoch,
        train_loss_l1_epoch, 
        valid_loss_l1_epoch
      )
    )
  else
    laia.log.info(
      string.format(
        '%d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f',
        epoch,
        train_loss_G_epoch,
        valid_loss_G_epoch,
        train_loss_D_epoch,
        valid_loss_D_epoch,
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

if output_progress_file ~= nil then output_progress_file:close() end