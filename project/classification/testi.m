training = importdata('classification_dataset_training.csv');
data = training.data;
covariates = data(:, 2:end-1);
variates = data(:, end);
poscov = covariates(variates == 1, :);
negcov = covariates(variates == 0, :);
poscovsum = sum(poscov);
negcovsum = sum(negcov);

subplot(221)
plot(1:length(poscovsum), poscovsum, 'kx')
title('pos')
subplot(223)
plot(1:length(negcovsum), negcovsum, 'kx')
title neg
subplot(224)
hold on
plot(1:length(poscovsum), poscovsum, 'bx')
plot(1:length(negcovsum), negcovsum, 'rx')
plot(1:length(poscovsum + negcovsum), poscovsum + negcovsum, 'kx')
title both

indices = [];

norpos = poscovsum/length(variates(variates==1));
norneg = negcovsum/length(variates(variates==0));

for i = 1:length(poscovsum)
    if(norpos(i) < norneg(i))
        indices(end+1) = i;
    end
end