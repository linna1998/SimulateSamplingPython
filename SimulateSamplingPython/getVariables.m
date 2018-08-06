% now sometimes it has some bugs QAQ
% 2018.8.3 16:33 DK
% return a variable which suits
% beginning 0 ~ constraintsId - 1 constraints
function variables = getVariables(constraintsId, parameter_isEqual, variablesNum, variables_lower, variables_upper, parameter_result)
    disp(constraintsId);
    disp(parameter_isEqual);
    disp(variablesNum);
    disp(variables_lower);
    disp(variables_upper);
    disp(parameter_result);    
    variables = [];
    if (constraintsId == 0)
        % matlab starts from 1 !!!!        
         for j = 1:variablesNum
             rng('shuffle');  % set for the rand 
             temp = randi([variables_lower(j), variables_upper(j)], 1, 1);
             variables = [variables; temp];
         end  
         disp(variables);
         return;
    else
        % Randomly get a variable
        % which suits for 0 ~ constraintsId - 1 constraints
        
         % Objective function
        F = zeros(1, variablesNum);
        F = [F, 1];
        disp("after build F");
        
        % int variables
        V_NUM = variablesNum + 1;
        intcon = ones(1, V_NUM);
        disp("after build intcon");
        
        % Build Constraints
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        A = [A; F]; % Concatenating Matrices
        b = [b; 0];
                
        for k = 1:constraintsId
            temp_a = parameter_result(k, :);
            disp("temp_a");
            disp(temp_a);
            temp_b = temp_a(variablesNum);
            if (not (parameter_isEqual(k)))
                % add fi - t <= 0
                temp_a(variablesNum) = -1;
                A = [A; temp_a];
                b = [b; -temp_b];
            else
                % add fi == 0 [for the equation constraints
                temp_a(variablesNum) = 0;
                Aeq = [Aeq; temp_a];
                beq = [beq; -temp_b];
            end
        end
        disp("after build constraints");
        disp("A");
        disp(A);
        disp("b");
        disp(b);
        disp("Aeq");
        disp(Aeq);
        disp("beq");
        disp(beq);
        
        rng('shuffle');  % set for the rand
        for i = 1:variablesNum
            variables_lower(i) = round(variables_lower(i) * rand());
            variables_upper(i) = round(variables_upper(i) * rand());
        end
        variables_lower = [variables_lower, 0];
        variables_upper = [variables_upper, 0];
        x0 = zeros(1, V_NUM);
        options = optimoptions('intlinprog');
        % options.IntegerTolerance must be in [1e-6, 0.001]
        % options.IntegerTolerance = 1e-05;
        variables = intlinprog(F, intcon, A, b, Aeq, beq, double(variables_lower), double(variables_upper), x0, options);
    end
    return;
