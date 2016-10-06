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
                res = neurfun(z);
            end
        end
        
        % Trains the neural network.
        function [obj, valerr] = Train(obj, training_data, validation_data, epochs, mini_batch_size, eta, lambda)
            size(training_data)
            size(validation_data)
            n = size(training_data, 1);
            regfact = 1-lambda*eta/n;
            
            valerr = zeros(1, epochs);
            val_cov = validation_data(:, 1:end-obj.layer_sizes(end))';
            val_var = validation_data(:, end-obj.layer_sizes(end)+1:end)';
            for i = 1:epochs
                shuffled = shuffle(training_data);
                for j = 1:mini_batch_size:n-mini_batch_size
                    mini_batch = shuffled(j:j+mini_batch_size, :);
                    covariates = mini_batch(:, 1:end-obj.layer_sizes(end));
                    variates = mini_batch(:, end-obj.layer_sizes(end)+1:end);
                    [grad_weights, grad_biases] = Backpropagate(obj, covariates', variates');
                    for k = 1:obj.len
                        obj.biases{k}  = obj.biases{k}-eta*grad_biases{k};
                        obj.weights{k} = regfact*obj.weights{k}-eta*grad_weights{k};
                    end
                end
                valerr(i) = real(mean(cost(Feedforward(obj, val_cov), val_var)));
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
                act = neurfun(z);
                acts{i+1} = act;
            end
            grad_weights = cell(1, obj.len);
            grad_biases = cell(1, obj.len);
            
            % Backpropagate
            grad_biases{end} = dcost(acts{end}, expected).*dneurfun(zs{end});
            grad_weights{end} = grad_biases{end}*acts{end-1}'/n;
            
            for i = fliplr(1:obj.len-1)
                grad_biases{i} = (obj.weights{i+1}'*grad_biases{i+1}).*dneurfun(zs{i});
                grad_weights{i} = grad_biases{i}*acts{i}'/n;
            end
            
            for i = 1:obj.len
                grad_biases{i} = mean(grad_biases{i}, 2);
            end
            
        end
        
    end
    
end

function res = neurfun(z)
    res = max(0, z);
end

% Is a derivative of neurfun function. Duh.
function res = dneurfun(z)
    %res = neurfun(z).*(1-neurfun(z));
    res = z;
    res(res<0) = 0;
    res(res>0) = 1;
end

% Is the cost function. a is the calculated result, r is the
% expected result.
function res = cost(a, r)
    %res = -sum(r'.*log(a') + (1-r').*log(1-a'))/size(a,2);
    res = (a-r).^2;
end

function res = dcost(a, r)
    res = bsxfun(@minus,a,r);
end