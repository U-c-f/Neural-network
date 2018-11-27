clc;
clear all;
close all;

tic;

x_samples = linspace(0,2,10000); 
y_samples = 0.6*sin(pi*x_samples)+0.3*sin(3*pi*x_samples)+0.1*sin(5*pi*x_samples)+0.05*sin(7*pi*x_samples);

epoch_num = input('Number of epochs :');
N = input('Total number of training examples per epoch :');
eta = input('Network learning rate :');
alpha = input('Alpha :');
layers_num = (input('Number of hidden layers :'))+1; 

for i=1:1:layers_num-1 
    
    layer(i).neurone_num  = input(strcat('Number of neurones in hidden layer (',int2str(i),') is :'));
    
    if (i==1)
        layer(i).threshold   = randn(layer(i).neurone_num,1);
        layer(i).weight = randn(layer(i).neurone_num,1);
        layer(i).previous_weight = zeros(layer(i).neurone_num,1);
    else
        layer(i).threshold   = randn(layer(i).neurone_num,1);
        layer(i).weight = randn(layer(i).neurone_num,layer(i-1).neurone_num); 
        layer(i).previous_weight = zeros(layer(i).neurone_num,layer(i-1).neurone_num);
        
    end 
    layer(i).activation_fct = 'tansig';
end

layer(layers_num).neurone_num  = 1;          
layer(layers_num).threshold = randn(1,1);
layer(layers_num).weight = randn(1,layer(layers_num - 1).neurone_num);
layer(layers_num).previous_weight = zeros(1,layer(layers_num - 1).neurone_num);
layer(layers_num).activation_fct = 'purelin';

clc;
display('Learning process has started');
timer=tic; 

for e = 1:1:epoch_num
    
    for n=1:1:N
        
       display(strcat(num2str(n/N*100),'% completed'));
      
       if(e==1) 
           index = randi(length(y_samples),1,1); 
           current_x = x_samples(index); 
           y_desired = 0.6*sin(pi*current_x)+0.3*sin(3*pi*current_x)+0.1*sin(5*pi*current_x)+0.05*sin(7*pi*current_x);
       else 
           copy_saved = saved(:,:);
           index = randi(length(y_samples),1,1); 
           current_x = copy_saved(index);
           y_desired = 0.6*sin(pi*current_x)+0.3*sin(3*pi*current_x)+0.1*sin(5*pi*current_x)+0.05*sin(7*pi*current_x);
       end
     
        for i=1:1:layers_num 
            if (i==1)
                layer(i).Vk = feval(layer(i).activation_fct,layer(i).weight*current_x+layer(i).threshold);
            else
                layer(i).Vk = feval(layer(i).activation_fct,layer(i).weight*layer(i-1).Vk+layer(i).threshold);
            end
        end
     
        y_actual = layer(layers_num).Vk;
        
        error(:,n) = y_desired - y_actual; 
       
        layer(layers_num).phiPrime=feval('dpurelin',1,y_actual); 
        layer(layers_num).delta=-2*diag(layer(layers_num).phiPrime)*error(:,n);
        
        for i=layers_num-1:-1:1 
            layer(i).phiPrime=feval('dtansig',1,layer(i).Vk);
            layer(i).delta = diag(layer(i).phiPrime)*layer(i+1).weight'*layer(i+1).delta; %% all hidden layers : compute delta
        end
     
        for i=1:1:layers_num 
            layer(i).threshold = layer(i).threshold-eta.*layer(i).delta;
            layer(i).previous_weight = layer(i).weight; 
      
            if (i~=1)
                layer(i).weight = layer(i).weight-eta.*layer(i).delta*layer(i-1).Vk' -alpha.*(layer(i).weight - layer(i).previous_weight)  ;
            else
                layer(i).weight = layer(i).weight-eta.*layer(i).delta*current_x'-alpha.*(layer(i).weight-layer(i).previous_weight) ;
            end
        end
        
        if (e==1)
             clc; 
             saved(n) = current_x;
             saved(randperm(length(saved)));
        else  
             clc;
             saved(randperm(length(saved))); 
        end
       
    end
    
    err_epoch(e,:) = error; 
    
end

clc; 
display(strcat('Learning process finished in (',num2str(toc(timer)),') seconds')); 
pause(0.5); 

Eav = 0;
for e=1:1:epoch_num
    
    err_squared = err_epoch(e,:).^2;
    
    sum_err_squared = 0;
    
    for i=1:1:length(err_squared)
        
        sum_err_squared = sum_err_squared + err_squared(i);
        
    end
    
    Eav = Eav + sum_err_squared;
    
end
Eav = Eav /(2*N);

display(strcat('Error on all epochs Eav = ',num2str(Eav)));
pause(5);

for i=1:1:length(x_samples)
    current_x=x_samples(i); 
    for j=1:1:layers_num 
        if (j==1)
            layer(j).Vk = feval(layer(j).activation_fct,layer(j).weight*current_x+layer(j).threshold);
        else
            layer(j).Vk = feval(layer(j).activation_fct,layer(j).weight*layer(j-1).Vk+layer(j).threshold);
        end
    end
    calculated_y(i)=layer(layers_num).Vk; 
end

figure(1);
plot(x_samples,y_samples,'b',x_samples,calculated_y,'g');
legend('show','Actual','Approximated');
xlabel('x');
ylabel('y');

pause(2);

while(1)

     clc;
     test_x = input('Enter a random x value between 0 and 2:');
     correct_y = 0.6*sin(pi*test_x)+0.3*sin(3*pi*test_x)+0.1*sin(5*pi*test_x)+0.05*sin(7*pi*test_x)

     if isempty(test_x)
         clc;
         break; 
     end

     for j=1:1:layers_num 

            if (j==1)
                layer(j).Vk = feval(layer(j).activation_fct,layer(j).weight*test_x+layer(j).threshold);
            else
                layer(j).Vk = feval(layer(j).activation_fct,layer(j).weight*layer(j-1).Vk+layer(j).threshold);
            end
            obtained_y = layer(layers_num).Vk;
     end

     display(strcat('The correct y is : (',num2str(correct_y),')'));
     pause(2);
     display(strcat('The obtained y is : (',num2str(obtained_y),')'));
     pause(10);
 
end 
