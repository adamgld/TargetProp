require 'nn'

local noise_injection = torch.class('nn.noise_injection', 'nn.Module')

function noise_injection:updateOutput(input)
  self.output:resizeAs(input):copy(input)
  -- ...something here...
  self.noise = torch.Tensor()
  self.noise:resizeAs(input)
  self.noise:normal(0,0.2)
  self.output:add(self.noise)
  return self.output
end

function noise_injection:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput -- no copy
  return self.gradInput
end
