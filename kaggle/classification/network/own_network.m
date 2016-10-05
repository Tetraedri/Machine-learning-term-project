function network = own_network(layer_sizes)
    biases = cell(1, length(layer_sizes)-1);
    weigths = cell(1, length(layer_sizes)-1);
    for i = 2:length(layer_sizes)
        biases{i-1} = rand(layer_sizes(i), 1)/layer_sizes(i-1);
        weigths{i-1} = rand(layer_sizes(i-1), layer_sizes(i))/layer_sizes(i-1);
    end
    network = struct('biases', biases, 'weights', weigths);
end

