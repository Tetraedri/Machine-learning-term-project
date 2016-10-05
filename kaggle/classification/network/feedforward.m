function res = feedforward(input, net)
    res = input;
    for i = 1:length(net)
        res = sigmoid(input(i).weigths*res-net(i).biases);
    end
end