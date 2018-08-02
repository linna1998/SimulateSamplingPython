function x = getLinear(A, B)
    disp(A);
    disp(B);
    % x = A \ B;
    f = zeros(size(A,1),1);
    x = intlinprog(f,1:size(A,1),[A;-A],[B;-B]);
end