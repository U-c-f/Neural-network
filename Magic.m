%%%%%%%%%%%%%%%% Start with these values %%%%%%%%%%%%%%%%

% 1 epoch
% 20 000 training examples
% eta = 0.001
% alpha = 0.0001
% 4 hidden layers
% 30 neurons each

%%%%%%%%%%%%%%%% Change parameters as you see fit %%%%%%%%%%%%%%%%

clc;
clear all;
close all;

tic; % to time the learning process

x_samples = linspace(0,2,1000); %% taking only one period of the function and 1000 samples
y_samples = 0.6*sin(pi*x_samples)+0.3*sin(3*pi*x_samples)+0.1*sin(5*pi*x_samples)+0.05*sin(7*pi*x_samples);

epoch_num = input('Number of epochs :');
N = input('Total number of training examples per epoch :');
eta = input('Network learning rate :');
alpha = input('Alpha :');
layers_num = (input('Number of hidden layers :'))+1;  % layers_num = nb of hidden layers + output layer

%%% initialize parameters for hidden layers
 
for i=1:1:layers_num-1 
    
    layer(i).neurone_num  = input(strcat('Number of neurones in hidden layer (',int2str(i),') is :'));
    
    if (i==1) %% first hidden layer ==> layer before is one neurone
        layer(i).threshold   = randn(layer(i).neurone_num,1); %% column vector of thresholds
        layer(i).weight = randn(layer(i).neurone_num,1); %% matrix of weights of size mxn where m input
        layer(i).previous_weight = zeros(layer(i).neurone_num,1); %% previous_weight is used to keep the weights of the current iteration for future use in weight adjustment
    else
        layer(i).threshold   = randn(layer(i).neurone_num,1);
        layer(i).weight = randn(layer(i).neurone_num,layer(i-1).neurone_num); 
        layer(i).previous_weight = zeros(layer(i).neurone_num,layer(i-1).neurone_num);
        
    end 
    layer(i).activation_fct = 'tansig';
end

%%% initalize parameters for output layer %%%

layer(layers_num).neurone_num  = 1;          
layer(layers_num).threshold = randn(1,1);
layer(layers_num).weight = randn(1,layer(layers_num - 1).neurone_num);
layer(layers_num).previous_weight = zeros(1,layer(layers_num - 1).neurone_num); %% previous_weight is used to keep the weights of the current iteration for future use in alpha
layer(layers_num).activation_fct = 'purelin';

%%%%%%%%%%%%%%%%%%%%% Training start %%%%%%%%%%%%%%%%%%%%%

clc;
display('Learning process has started');
timer=tic; % timer starts


for e = 1:1:epoch_num
    
    for n=1:1:N
       display(strcat(num2str(n/N*100),'% completed'));
        
        %%% picking a random x value from the set of samples
        
       if(e==1) %% if first epoch pick a random index from x and compute y
           index = randi(length(y_samples),1,1); %%choosing a random index in linspace set - UNIFORM discrete distribution.
           current_x = x_samples(index); %% calculating the corresponding value of the picked index (value is between 0 and 2)
           y_desired = 0.6*sin(pi*current_x)+0.3*sin(3*pi*current_x)+0.1*sin(5*pi*current_x)+0.05*sin(7*pi*current_x);
       else %% 2nd or more epoch uses the same training examples after being shuffled
           copy_saved = saved(:,:); %% keeping a copy of the saved values so we can shuffle them and use them in another epoch
           index = randi(length(y_samples),1,1); 
           current_x = copy_saved(index);
           y_desired = 0.6*sin(pi*current_x)+0.3*sin(3*pi*current_x)+0.1*sin(5*pi*current_x)+0.05*sin(7*pi*current_x);
       end
        
        %%%% direct calculation %%%
        
        for i=1:1:layers_num %% compute Vk ==> output of each neuron
            if (i==1)
                layer(i).Vk = feval(layer(i).activation_fct,layer(i).weight*current_x+layer(i).threshold);
            else
                layer(i).Vk = feval(layer(i).activation_fct,layer(i).weight*layer(i-1).Vk+layer(i).threshold);
            end
        end
     
        y_actual = layer(layers_num).Vk;
        
        error(:,n) = y_desired - y_actual; %% error vector (one column at a time)
        
        
        %%%%%%%%% Backpropagation %%%%%%%%%
        
       
        layer(layers_num).phiPrime=feval('dpurelin',1,y_actual);  %compute phi' for output layer to compute delta and start backprop
        layer(layers_num).delta=-2*diag(layer(layers_num).phiPrime)*error(:,n);%Compute delta of output layer (use diag for dimensions to agree in matrix product)
        
        % Go backwards and compute phi' and delta for each hidden layer

        for i=layers_num-1:-1:1 %% starting at the last hidden layer (layers_num-1)
            layer(i).phiPrime=feval('dtansig',1,layer(i).Vk);
            layer(i).delta = diag(layer(i).phiPrime)*layer(i+1).weight'*layer(i+1).delta; %% all hidden layers : compute delta
        end
        
         
        % Adjusting weights and thresholds
        
        for i=1:1:layers_num 
            layer(i).threshold = layer(i).threshold-eta.*layer(i).delta;
            layer(i).previous_weight = layer(i).weight; %% saving the previous weights 
      
            if (i~=1)
                layer(i).weight = layer(i).weight-eta.*layer(i).delta*layer(i-1).Vk' -alpha.*(layer(i).weight - layer(i).previous_weight)  ;
            else
                layer(i).weight = layer(i).weight-eta.*layer(i).delta*current_x'-alpha.*(layer(i).weight- layer(i).previous_weight) ;
            end
        end
        
        if (e==1) %% first epoch ==> save each current_x to use them in future epochs (after permutation)
             clc; 
             saved(n) = current_x;
             saved(randperm(length(saved)));
        else %% epoch e >= 2 ==> just do permutation 
             clc;
             saved(randperm(length(saved))); 
        end
       
    end
    
    err_epoch(e,:) = error; %%% keeping the error vector of each epoch in a matrix for the next calculations
    
end

%%%%%%%%%%%%%%%%%%%%% Training end %%%%%%%%%%%%%%%%%%%%%

clc; 
display(strcat('Learning process finished in (',num2str(toc(timer)),') seconds')); %% timer stops
pause(0.5); 

%%%%%%% Calculation of the error on all epochs Eav, and displaying the results %%%%%%%

Eav = 0;

for e=1:1:epoch_num
    
    err_squared = err_epoch(e,:).^2;
    
    sum_err_squared = 0;
    
    for i=1:1:length(err_squared)
        
        sum_err_squared = sum_err_squared + err_squared(i);
        
    end
    
    Eav = Eav + sum_err_squared;
    
end

display(strcat('Error on all epochs Eav = ',num2str(Eav)));

%%%%%%% Calculation of  Eav end %%%%%%%




% Weights are saved after training : compute the values of y after theweights have been adjusted

for i=1:1:length(x_samples)
    current_x=x_samples(i); %% taking each sample x at a time and calculate corresponding y
    %%%% redo direct calculation with the values of the weights 
    for j=1:1:layers_num 
        if (j==1)
            layer(j).Vk = feval(layer(j).activation_fct,layer(j).weight*current_x+layer(j).threshold);
        else
            layer(j).Vk = feval(layer(j).activation_fct,layer(j).weight*layer(j-1).Vk+layer(j).threshold);
        end
    end
    calculated_y(i)=layer(layers_num).Vk; %% vector of calculated y by the network
end

%%%% plot %%%%

figure(1);
plot(x_samples,y_samples,'b',x_samples,calculated_y,'g');
xlabel('x');
ylabel('y');
