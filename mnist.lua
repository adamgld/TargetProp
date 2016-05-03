require 'torch'
require 'nn'
require 'math'
require 'dataset-mnist'
require 'gnuplot'

mnist.download()       -- download dataset if not already there

-- load dataset using dataset-mnist.lua into tensors (first dim of data/labels ranges over data)
local function load_dataset(train_or_test, count)
    -- Subsampling mask
    local mask = torch.Tensor(2,2)
    mask:fill(0.25)

    -- load
    local data
    if train_or_test == 'train' then
        data = mnist.loadTrainSet(count, {32, 32})
    else
        data = mnist.loadTestSet(count, {32, 32})
    end
	local datas = torch.Tensor(torch.LongStorage{count,1,16,16})
    for i=1,count do
 	local image = torch.conv2(data.data[i][1],mask)
   	local temp = torch.Tensor(image:storage(),1,torch.LongStorage{16,16},torch.LongStorage{62,2}):clone()

	datas[i][1] = temp

    end
	data.data = datas

    -- shuffle the dataset
    local shuffled_indices = torch.randperm(data.data:size(1)):long()
    -- creates a shuffled *copy*, with a new storage
    data.data = data.data:index(1, shuffled_indices):squeeze()
    data.labels = data.labels:index(1, shuffled_indices):squeeze()

    -- TODO: (optional) UNCOMMENT to display a training example
    -- for more, see torch gnuplot package documentation:
    -- https://github.com/torch/gnuplot#plotting-package-manual-with-gnuplot
    --local subsampl = torch.Tensor(image:storage(),1,torch.LongStorage{16,16},torch.LongStorage{62,2}):clone()
    -- gnuplot.imagesc(data.data[10])
    -- vectorize each 2D data point into 1D
    data.data = data.data:reshape(data.data:size(1), 16*16)

    print('--------------------------------')
    print(' loaded dataset "' .. train_or_test .. '"')
    print('inputs', data.data:size())
    print('targets', data.labels:size())
    print('--------------------------------')

    return data
end

--local 
train = load_dataset('train', 3000)
--local 
test = load_dataset('test', 1000)

