function gmres_experiment()
    matrices = {"cavity05", "cz2548", "epb3"};
    test_params = [false, true, true];
    ks = {
        [200],
        [100, 250, 500],
        [100, 250, 500]
    };
    tols = {
        [1e-5],
        [1e-5, 1e-7, 1e-9, 1e-11],
        [1e-5, 1e-7, 1e-9, 1e-11]
    };
    maxit = [1e2];

    for i = 1 : numel(matrices)

        load(sprintf("in/%s.mat", matrices{i}));

        data_file = fopen(sprintf("out/%s_data.txt", matrices{i}), "w");

        A = Problem.A;
        n = rows(A);
        solution = ones(n, 1);
        b = A * solution;
        [r, c] = find(A);

        fprintf(data_file, "n: %d\n", n);
        fprintf(data_file, "nnz: %d\n", nnz(A));
        fprintf(data_file, "band: %d\n", max(r - c));

        hf = figure();
        spy(A);
        print(hf, sprintf("out/%s_spy_A.png", matrices{i}), "-dpng");

        perm = symrcm(A);
        identity = speye(n, n);
        P = identity(perm, :);
        R = P * A * P';
        rb = P * b * P';

        hf = figure();
        spy(R);
        print(hf, sprintf("out/%s_spy_R.png", matrices{i}), "-dpng");

        used_k = 0;
        used_tol = .0;
        used_maxit = 0;

        # No preconditioning
        fprintf(data_file, "\nNo Preconditioning\n\n");

        [noprecond_x, noprecond_flag, noprecond_relres, noprecond_iter, noprecond_resvec] = [zeros(n, 1), -1, .0, zeros(1, 2), zeros(n, 2)];

        timer = clock();
        if test_params(i)
            [noprecond_x, noprecond_flag, noprecond_relres, noprecond_iter, noprecond_resvec, used_k, used_tol, used_maxit] = test_gmres_params(A, b, ks{i}, tols{i}, maxit);
        else
            used_k = ks{i}(1);
            used_tol = tols{i}(1);
            used_maxit = maxit(1);
            [noprecond_x, noprecond_flag, noprecond_relres, noprecond_iter, noprecond_resvec] = gmres(A, b, ks{i}(1), tols{i}(1), maxit(1));
        end
        elapsed_time = etime(clock(), timer);

        noprecond_iter_n = noprecond_iter(1, 1) * used_k + noprecond_iter(1, 2);

        fprintf(data_file, "Convergence Flag: %d\n", noprecond_flag);
        fprintf(data_file, "Iterations: %d\n", noprecond_iter_n);
        fprintf(data_file, "Solution Inf. Norm: %e\n", norm(noprecond_x, inf));
        fprintf(data_file, "Elapsed time: %ds\n", elapsed_time);

        # Only ILU(0) preconditioning
        fprintf(data_file, "\nILU(0) Preconditioning\n\n");

        timer = clock();
        opts.type = "nofill";
        [L, U] = ilu(A, opts);
        [ilu0_x, ilu0_flag, ilu0_relres, ilu0_iter, ilu0_resvec] = gmres(A, b, used_k, used_tol, used_maxit, L, U);
        elapsed_time = etime(clock(), timer);

        ilu0_iter_n = ilu0_iter(1, 1) * used_k + ilu0_iter(1, 2);

        fprintf(data_file, "Convergence Flag: %d\n", ilu0_flag);
        fprintf(data_file, "Iterations: %d\n", ilu0_iter_n);
        fprintf(data_file, "Solution Inf. Norm: %e\n", norm(ilu0_x, inf));
        fprintf(data_file, "nnz [L, U]: (%d, %d)\n", nnz(L), nnz(U));
        fprintf(data_file, "Elapsed time: %ds\n", elapsed_time);

        hf = figure();
        spy(L);
        print(hf, sprintf("out/%s_spy_L_ilu0.png", matrices{i}), "-dpng");
        hf = figure();
        spy(U);
        print(hf, sprintf("out/%s_spy_U_ilu0.png", matrices{i}), "-dpng");
        
        # ILU(0) preconditioning with line reordering
        fprintf(data_file, "\nILU(0) Preconditioning RCM\n\n");

        timer = clock();
        opts.type = "nofill";
        [L, U] = ilu(R, opts);
        [ilu0r_x, ilu0r_flag, ilu0r_relres, ilu0r_iter, ilu0r_resvec] = gmres(R, rb, used_k, used_tol, used_maxit, L, U);
        elapsed_time = etime(clock(), timer);

        ilu0r_iter_n = ilu0r_iter(1, 1) * used_k + ilu0r_iter(1, 2);

        fprintf(data_file, "Convergence Flag: %d\n", ilu0r_flag);
        fprintf(data_file, "Iterations: %d\n", ilu0r_iter_n);
        fprintf(data_file, "Solution Inf. Norm: %e\n", norm(ilu0r_x, inf));
        fprintf(data_file, "nnz [L, U]: (%d, %d)\n", nnz(L), nnz(U));
        fprintf(data_file, "Elapsed time: %ds\n", elapsed_time);

        hf = figure();
        spy(L);
        print(hf, sprintf("out/%s_spy_L_ilu0r.png", matrices{i}), "-dpng");
        hf = figure();
        spy(U);
        print(hf, sprintf("out/%s_spy_U_ilu0r.png", matrices{i}), "-dpng");

        # Only ILU crout preconditioning
        fprintf(data_file, "\nILU Crout Preconditioning\n\n");

        timer = clock();
        opts.type = "crout";
        opts.droptol = 1e-4;
        [L, U] = ilu(A, opts);
        [crout_x, crout_flag, crout_relres, crout_iter, crout_resvec] = gmres(A, b, used_k, used_tol, used_maxit, L, U);
        elapsed_time = etime(clock(), timer);

        crout_iter_n = crout_iter(1, 1) * used_k + crout_iter(1, 2);

        fprintf(data_file, "Convergence Flag: %d\n", crout_flag);
        fprintf(data_file, "Iterations: %d\n", crout_iter_n);
        fprintf(data_file, "Solution Inf. Norm: %e\n", norm(crout_x, inf));
        fprintf(data_file, "nnz [L, U]: (%d, %d)\n", nnz(L), nnz(U));
        fprintf(data_file, "Elapsed time: %ds\n", elapsed_time);

        hf = figure();
        spy(L);
        print(hf, sprintf("out/%s_spy_L_crout.png", matrices{i}), "-dpng");
        hf = figure();
        spy(U);
        print(hf, sprintf("out/%s_spy_U_crout.png", matrices{i}), "-dpng");

        # ILU crout preconditioning with line reordering
        fprintf(data_file, "\nILU Crout Preconditioning RCM\n\n");

        timer = clock();
        opts.type = "crout";
        opts.droptol = 1e-4;
        [L, U] = ilu(R, opts);
        [croutr_x, croutr_flag, croutr_relres, croutr_iter, croutr_resvec] = gmres(R, rb, used_k, used_tol, used_maxit, L, U);
        elapsed_time = etime(clock(), timer);

        croutr_iter_n = croutr_iter(1, 1) * used_k + croutr_iter(1, 2);

        fprintf(data_file, "Convergence Flag: %d\n", croutr_flag);
        fprintf(data_file, "Iterations: %d\n", croutr_iter_n);
        fprintf(data_file, "Solution Inf. Norm: %e\n", norm(croutr_x, inf));
        fprintf(data_file, "nnz [L, U]: (%d, %d)\n", nnz(L), nnz(U));
        fprintf(data_file, "Elapsed time: %ds\n", elapsed_time);
        fclose(data_file);

        hf = figure();
        spy(L);
        print(hf, sprintf("out/%s_spy_L_croutr.png", matrices{i}), "-dpng");
        hf = figure();
        spy(U);
        print(hf, sprintf("out/%s_spy_U_croutr.png", matrices{i}), "-dpng");

        max_iter = max([noprecond_iter_n, ilu0_iter_n, ilu0r_iter_n, crout_iter_n, croutr_iter_n]);

        adj_noprecond_resvec = zeros(max_iter, 1);
        for j = 1 : min(max_iter, numel(noprecond_resvec))
            adj_noprecond_resvec(j) = noprecond_resvec(j);
        end

        adj_ilu0_resvec = zeros(max_iter, 1);
        for j = 1 : min(max_iter, numel(ilu0_resvec))
            adj_ilu0_resvec(j) = ilu0_resvec(j);
        end

        adj_ilu0r_resvec = zeros(max_iter, 1);
        for j = 1 : min(max_iter, numel(ilu0r_resvec))
            adj_ilu0r_resvec(j) = ilu0r_resvec(j);
        end

        adj_crout_resvec = zeros(max_iter, 1);
        for j = 1 : min(max_iter, numel(crout_resvec))
            adj_crout_resvec(j) = crout_resvec(j);
        end

        adj_croutr_resvec = zeros(max_iter, 1);
        for j = 1 : min(max_iter, numel(croutr_resvec))
            adj_croutr_resvec(j) = croutr_resvec(j);
        end

        hf = figure();
        plot(1 : max_iter, log(adj_noprecond_resvec), "k");
        hold on;
        plot(1 : max_iter, log(adj_ilu0_resvec), "r");
        hold on;
        plot(1 : max_iter, log(adj_ilu0r_resvec), "g");
        hold on;
        plot(1 : max_iter, log(adj_crout_resvec), "b");
        hold on;
        plot(1 : max_iter, log(adj_croutr_resvec), "y");
        legend("No Prec.", "ILU(0)", "ILU(0) RCM", "Crout", "Crout RCM");
        print(hf, sprintf("out/%s_res_iter.png", matrices{i}), "-dpng");

    end
end
