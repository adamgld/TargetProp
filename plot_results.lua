scale = function(x,down,up)
        x_min = torch.min(x)
        x_max = torch.max(x)
        x:add(-x_min):mul(1/(torch.max(x)-torch.min(x)))
        x:mul(up-down):add(down)
        return x
end

plot = function()
	print("The image is a digit ", test.labels[10]-1)
	input = test.data[10]
	inimage = input:clone():resize(16,16)
	noisy = n1:forward(input)
	noisyimage = noisy:clone():resize(16,16)
	reconstr = l2:forward(l1:forward(input)):resize(16,16)

	gnuplot.imagesc(torch.cat({scale(inimage,0,256),scale(noisyimage,0,256),scale(reconstr,0,256)}))

	w,_ = model:parameters()
	bar = torch.Tensor(16)
	bar:fill(1.0)

	image.save("weight_examples.pgm",torch.cat({scale(w[1][1]:resize(16,16),0.0,1.0),bar,scale(w[1][2]:resize(16,16),0.0,1.0),bar,scale(w[1][3]:resize(16,16),0.0,1.0),bar,scale(w[1][4]:resize(16,16),0.0,1.0),bar,scale(w[1][5]:resize(16,16),0.0,1.0)}),5)
end

