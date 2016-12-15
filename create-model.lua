#!/usr/bin/env th

require 'laia'
require 'models'

local parser = laia.argparse(){
  name = 'create_model',
  description = 'Create a model for pix2pix composed by a ' ..
    'discriminator network plus a generator network (GAN arquitecture).'
}

parser:option(
  '--discriminator_type',
  'Type of network architecture to use for the discriminator network, ' ..
  'valid types are basic and n_layers',
  'basic', {basic = 'basic', n_layers = 'n_layers'})
  :argname('<type>')

parser:option(
  '--generator_type',
  'Type of network architecture to use for the discriminator network, ' ..
  'valid types are encoder_decoder and unet',
  'unet', {encoder_decoder = 'encoder_decoder', unet = 'unet'})
  :argname('<type>')

parser:option(
  '--conditional_discriminator',
  'Use conditional discriminator.',
  {true}, toboolean)
  :argname('<bool>')

parser:option(
  '--num_units_discriminator',
  'Number of units in first layer of the discriminator, n > 0.', 64, laia.toint)
  :argname('<n>')
  :gt(0)

parser:option(
  '--num_units_generator',
  'Number of units in first layer of the generator, n > 0.', 64, laia.toint)
  :argname('<n>')
  :gt(0)

parser:option(
  '--num_layers_discriminator',
  'Number of layers of the num_layers_discriminator, ' .. 
  'only used if discriminator_type == \'n_layers\'.', 0, laia.toint)
  :argname('<n>')
  :gt(0)

parser:option(
  '--seed -s', 'Seed for random numbers generation.',
  0x012345, laia.toint)

-- Arguments
parser:argument(
  'input_channels', 'Number of channels of the input/output images.')
  :convert(laia.toint)
  :gt(0)

parser:argument(
  'output_file', 'Output file to store the model')

-- Register logging options
laia.log.registerOptions(parser)

local opt = parser:parse()

-- Initialize random seeds
laia.manualSeed(opt.seed)

local function initializeWeights(m)
  local name = torch.type(m)
  if name:find('Convolution') then
    m.weight:normal(0.0, 0.02)
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
  end
end

function createGenerator(input_channels, output_channels, num_units_generator)
  local gen = nil
  if opt.generator_type == 'encoder_decoder' then 
    gen = defineG_encoder_decoder(input_channels, output_channels, num_units_generator)
  elseif opt.generator_type == 'unet' then 
    gen = defineG_unet(input_channels, output_channels, num_units_generator)
  else 
    laia.log.error("Unsupported generator type.")
  end

  gen:apply(initializeWeights)

  return gen
end

function createDiscriminator(input_channels, output_channels, num_units_discriminator)  
  local dis = nil
  if not opt.conditional_discriminator then
    input_channels = 0 -- only penalizes structure in output channels
  end

  if opt.discriminator_type == 'basic' then 
    dis = defineD_basic(input_channels, output_channels, num_units_discriminator)
  elseif opt.discriminator_type == 'n_layers' then 
    dis = defineD_n_layers(input_channels, output_channels, num_units_discriminator, opt.num_layers_discriminator)
  else 
    laia.log.error("Unsupported discriminator type.")
  end
  
  dis:apply(initializeWeights)
  
  return dis
end

generator = createGenerator(opt.input_channels, opt.input_channels, opt.num_units_generator)
discriminator = createDiscriminator(opt.input_channels, opt.input_channels, opt.num_units_discriminator)

-- conver to cpu
generator:float()
discriminator:float()

-- Save models to disk
local checkpoint = {
  best = {
    modelG = generator,
    modelD = discriminator
  },
  last = {
    modelG = generator,
    modelD = discriminator
  },
  model_opt = opt 
}

local pg, _ = generator:getParameters()
local pd, _ = discriminator:getParameters()
laia.log.info('\n' .. generator:__tostring__())
laia.log.info('\n' .. discriminator:__tostring__())

laia.log.info('Saved models to %q', opt.output_file)

torch.save(opt.output_file, checkpoint)

