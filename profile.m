w_file = 'weights.mat';
g_file = 'grad.mat';

% [weights, mask_weights, grad_weights] 
raw_w = load(char(w_file));
raw_g = load(char(g_file));


weights_val = raw_w.weights .* (raw_w.mask==0);
gradient_val = raw_g.gradients .* (raw_w.mask==0);

weights_org = raw_w.weights;
gradient_org = raw_g.gradients;

figure
subplot(2,1,1)
plot(weights_org(1:200),'*');
hold on 
plot(gradient_org(1:200),'*');
weights_addr = weights_val~=0;
gradient_addr = gradient_val~=0;
weights_val = weights_val(weights_addr);
grandient_val = gradient_val(gradient_addr);
title('All weights and grads');

xlabel('Weights Id','FontSize',20);
ylabel('Value','FontSize',20);
set(gca, 'fontsize',18);


subplot(2,1,2)
plot(weights_val(1:200),'*');
hold on 
plot(grandient_val(1:200),'*');
title('Weights and grads that have been masked out');
xlabel('Number of Nodes','FontSize',20);
ylabel('Memory Size','FontSize',20);
set(gca,'fontsize',18);

