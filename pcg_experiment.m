function pcg_experiment()
    matrices = {"mesh3em5", "662_bus", "pdb1HYS", "Dubcova3"};
    tols = [1e-10, 1e-10, 1e-7, 1e-10];
    maxit = [1e5];

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
        rb = P * b;

        hf = figure();
        spy(R);
        print(hf, sprintf("out/%s_spy_R.png", matrices{i}), "-dpng");

        used_k = 0;
        used_tol = .0;
        used_maxit = 0;

        # No preconditioning
        fprintf(data_file, "\nNo Preconditioning\n\n");

        timer = clock();
        [noprecond_x, noprecond_flag, noprecond_relres, noprecond_iter, noprecond_resvec] = pcg(A, b, tols(i), maxit(1));
        elapsed_time = etime(clock(), timer);

        fprintf(data_file, "Convergence Flag: %d\n", noprecond_flag);
        fprintf(data_file, "Iterations: %d\n", noprecond_iter);
        fprintf(data_file, "Solution Inf. Norm: %e\n", norm(noprecond_x, inf));
        fprintf(data_file, "Elapsed time: %ds\n", elapsed_time);

        # Only ILU(0) preconditioning
        fprintf(data_file, "\nICC(0) Preconditioning\n\n");

        timer = clock();
        opts.type = "nofill";
        L = ichol(A, opts);
        [icc0_x, icc0_flag, icc0_relres, icc0_iter, icc0_resvec] = pcg(A, b, tols(i), maxit(1), L * L');
        elapsed_time = etime(clock(), timer);

        fprintf(data_file, "Convergence Flag: %d\n", icc0_flag);
        fprintf(data_file, "Iterations: %d\n", icc0_iter);
        fprintf(data_file, "Solution Inf. Norm: %e\n", norm(icc0_x, inf));
        fprintf(data_file, "nnz L: %d\n", nnz(L));
        fprintf(data_file, "Elapsed time: %ds\n", elapsed_time);

        hf = figure();
        spy(L);
        print(hf, sprintf("out/%s_spy_L_icc0.png", matrices{i}), "-dpng");
        
        # ILU(0) preconditioning with line reordering
        fprintf(data_file, "\nICC(0) Preconditioning RCM\n\n");

        timer = clock();
        opts.type = "nofill";
        L = ichol(R, opts);
        [icc0r_x, icc0r_flag, icc0r_relres, icc0r_iter, icc0r_resvec] = pcg(A, b, tols(i), maxit(1), L * L');
        elapsed_time = etime(clock(), timer);


        fprintf(data_file, "Convergence Flag: %d\n", icc0r_flag);
        fprintf(data_file, "Iterations: %d\n", icc0r_iter);
        fprintf(data_file, "Solution Inf. Norm: %e\n", norm(icc0r_x, inf));
        fprintf(data_file, "nnz L: %d\n", nnz(L));
        fprintf(data_file, "Elapsed time: %ds\n", elapsed_time);

        hf = figure();
        spy(L);
        print(hf, sprintf("out/%s_spy_L_icc0r.png", matrices{i}), "-dpng");

        # Only ILU crout preconditioning
        fprintf(data_file, "\nICT Preconditioning\n\n");

        timer = clock();
        opts.type = "ict";
        opts.droptol = 1e-4;
        L = ichol(A, opts);
        [ict_x, ict_flag, ict_relres, ict_iter, ict_resvec] = pcg(A, b, tols(i), maxit(1), L * L');
        elapsed_time = etime(clock(), timer);

        fprintf(data_file, "Convergence Flag: %d\n", ict_flag);
        fprintf(data_file, "Iterations: %d\n", ict_iter);
        fprintf(data_file, "Solution Inf. Norm: %e\n", norm(ict_x, inf));
        fprintf(data_file, "nnz L: %d\n", nnz(L));
        fprintf(data_file, "Elapsed time: %ds\n", elapsed_time);

        hf = figure();
        spy(L);
        print(hf, sprintf("out/%s_spy_L_ict.png", matrices{i}), "-dpng");

        # ILU crout preconditioning with line reordering
        fprintf(data_file, "\nICT Preconditioning RCM\n\n");

        timer = clock();
        opts.type = "ict";
        opts.droptol = 1e-4;
        L = ichol(R, opts);
        [ictr_x, ictr_flag, ictr_relres, ictr_iter, ictr_resvec] = gmres(R, rb, used_k, used_tol, used_maxit, L, U);
        elapsed_time = etime(clock(), timer);

        fprintf(data_file, "Convergence Flag: %d\n", ictr_flag);
        fprintf(data_file, "Iterations: %d\n", ictr_iter);
        fprintf(data_file, "Solution Inf. Norm: %e\n", norm(ictr_x, inf));
        fprintf(data_file, "nnz L: %d\n", nnz(L));
        fprintf(data_file, "Elapsed time: %ds\n", elapsed_time);
        fclose(data_file);

        hf = figure();
        spy(L);
        print(hf, sprintf("out/%s_spy_L_ictr.png", matrices{i}), "-dpng");

        max_iter = max([noprecond_iter, icc0_iter, icc0r_iter, ict_iter, ictr_iter]);

        adj_noprecond_resvec = zeros(max_iter, 1);
        for j = 1 : numel(noprecond_resvec)
            adj_noprecond_resvec(j) = noprecond_resvec(j);
        end

        adj_icc0_resvec = zeros(max_iter, 1);
        for j = 1 : numel(icc0_resvec)
            adj_icc0_resvec(j) = icc0_resvec(j);
        end

        adj_icc0r_resvec = zeros(max_iter, 1);
        for j = 1 : numel(icc0r_resvec)
            adj_icc0r_resvec(j) = icc0r_resvec(j);
        end

        adj_ict_resvec = zeros(max_iter, 1);
        for j = 1 : numel(ict_resvec)
            adj_ict_resvec(j) = ict_resvec(j);
        end

        adj_ictr_resvec = zeros(max_iter, 1);
        for j = 1 : numel(ictr_resvec)
            adj_ictr_resvec(j) = ictr_resvec(j);
        end

        hf = figure();
        plot(1 : max_iter, adj_noprecond_resvec, "k", 1 : max_iter, adj_icc0_resvec, "r", 1 : max_iter, adj_icc0r_resvec, "g", 1 : max_iter, adj_ict_resvec, "b", 1 : max_iter, adj_ictr_resvec, "y");
        legend("No Prec.", "ICC(0)", "ICC(0) RCM", "ICT", "ICT RCM");
        print(hf, sprintf("out/%s_res_iter.png", matrices{i}), "-dpng");

    end
end
