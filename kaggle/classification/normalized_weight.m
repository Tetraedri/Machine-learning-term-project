function result = normalized_weight
    weigths = train;
    result = test(weigths);
    
end

function weights = train
    training = importdata('classification_dataset_training.csv');
    data = training.data;
    covariates = data(:, 2:end-1);
    variates = data(:, end);
    poscov = covariates(variates == 1, :);
    negcov = covariates(variates == 0, :);
    poscovsum = sum(poscov);
    negcovsum = sum(negcov);

    norpos = poscovsum/length(variates(variates==1));
    norneg = negcovsum/length(variates(variates==0));

    weights = norpos-norneg;
end

function result = test(weights)
    testdata = importdata('classification_dataset_testing.csv');
    data = testdata.data;
    covariates = data(:, 2:end);
    result = zeros(length(covariates), 1);
    for i = 1:length(covariates)
        a = covariates(i,:);
        result_data = covariates(i,:).*weights;
        positives = result_data(result_data>0);
        negatives = result_data(result_data<0);
        pos = sum(positives);
        neg = sum(negatives);
        res = pos+abs(neg);
        result(i) = max(0.05, min(pos/res, 0.95));
    end
end