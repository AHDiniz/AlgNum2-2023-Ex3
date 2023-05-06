function [best_x, best_flag, best_relres, best_iter, best_resvec, best_k, best_tol, best_maxit] = test_gmres_params(A, b, ks, tols, maxits)

    inputs = cell(numel(ks) * numel(tols) * numel(maxits), 3);
    for i = 1 : numel(ks)
        for j = 1 : numel(tols)
            for l = 1 : numel(maxits)
                index = (i - 1) * numel(tols) + (j - 1) * numel(maxits) + l;
                inputs{index,1} = ks(i);
                inputs{index,2} = tols(j);
                inputs{index,3} = maxits(l);
            endfor
        endfor
    endfor

    best_sol_diff = inf;
    for i = 1 : numel(ks) * numel(tols) * numel(maxits)
        k = inputs{i,1};
        tol = inputs{i,2};
        maxit = inputs{i,3};
        printf("k = %d, tol = %e, maxit = %d\n", k, tol, maxit);
        tic
        [x, cflag, relres, iter, resvec] = gmres(A, b, k, tol, maxit);
        toc
        sol_diff = abs(norm(x, inf) - 1);
        if sol_diff < best_sol_diff
            best_x = x;
            best_flag = cflag;
            best_relres = relres;
            best_iter = iter;
            best_resvec = resvec;
            best_k = k;
            best_tol = tol;
            best_maxit = maxit;
            best_sol_diff = sol_diff;
        endif
    endfor
end
