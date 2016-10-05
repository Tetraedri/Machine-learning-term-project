classdef OwnNetwork
    properties
        biases
        weights
        layer_sizes
        len
    end
    
    methods
        
        % Initializes the network.
        function obj = OwnNetwork(layer_sizes)
            obj.layer_sizes = layer_sizes;
            obj.len = length(layer_sizes)-1;
            obj.biases = cell(1, obj.len);
            obj.weights = cell(1, obj.len);
            for i = 2:length(layer_sizes)
                obj.biases{i-1} = rand(layer_sizes(i), 1)/layer_sizes(i-1);
                obj.weights{i-1} = rand(layer_sizes(i), layer_sizes(i-1))/layer_sizes(i-1);
            end
        end
        
        % Outputs the result of an input
        function res = Feedforward(obj, input)
            res = input;
            for i = 1:obj.len
                z = bsxfun(@plus, obj.weights{i}*res, obj.biases{i});
                res = sigmoid(z);
            end
        end
        
        % Trains the neural network.
        function obj = Train(obj, training_data, epochs, mini_batch_size, eta)
            n = size(training_data, 1);
            for i = 1:epochs
                shuffled = shuffle(training_data);
                for j = 1:mini_batch_size:n-1
                    mini_batch = shuffled(j:j+mini_batch_size, :);
                    covariates = mini_batch(:, 1:end-1);
                    variates = mini_batch(:, end);
                    [grad_weights, grad_biases] = Backpropagate(obj, covariates', variates');
                    for k = 1:obj.len
                        obj.biases{k}  = obj.biases{k}-eta*grad_biases{k};
                        obj.weights{k} = obj.weights{k}-eta*grad_weights{k};
                    end
                end
            end
        end
        
        % Backpropagation. Here happens the optimization.
        function [grad_weights, grad_biases] = Backpropagate(obj, input, expected)
            n = size(expected, 2);
            
            acts = cell(1, obj.len+1);
            acts{1} = input;
            
            zs = cell(1, obj.len);
            
            % Feedforward
            for i = 1:obj.len             
                z = bsxfun(@plus, obj.weights{i}*acts{i}, obj.biases{i});
                zs{i} = z;
                act = sigmoid(z);
                acts{i+1} = act;
            end
            grad_weights = cell(1, obj.len);
            grad_biases = cell(1, obj.len);
            
            % Backpropagate
            grad_biases{end} = dcost(acts{end}, expected).*dsigmoid(zs{end});
            grad_weights{end} = grad_biases{end}*acts{end-1}'/n;
            
            for i = fliplr(1:obj.len-1)
                grad_biases{i} = (obj.weights{i+1}'*grad_biases{i+1}).*dsigmoid(zs{i});
                grad_weights{i} = grad_biases{i}*acts{i}'/n;
            end
            
            for i = 1:obj.len
                grad_biases{i} = mean(grad_biases{i}, 2);
            end
            
        end
        
    end
    
end

% Is a derivative of sigmoid function. Duh.
function res = dsigmoid(z)
    res = sigmoid(z).*(1-sigmoid(z));
end

% Is the cost function. a is the calculated result, r is the
% expected result.
function res = cost(a, r)
    res = -sum(r.*log(a) + (1-r).*log(1-a))/size(a,2);
end

function res = dcost(a, r)
    res = bsxfun(@minus,a,r);
end