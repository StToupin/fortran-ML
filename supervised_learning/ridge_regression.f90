subroutine ridge_regression_train(n, d, X_train, y_train, lambda, w)
    implicit none
    integer, intent(in) :: n, d
    double precision, intent(in) :: X_train(n,d), y_train(n), lambda
    double precision, intent(out) :: w(d+1)
    
    double precision :: Xones(n,d+1), invSigma(d+1,d+1)
    integer :: i, info, ipiv(d+1)
    
    Xones(:,1:d) = X_train
    Xones(:,d+1) = 1.d0

    call dgemm('T', 'N', d+1, d+1, n, 1.d0, Xones, n, Xones, n, 0.d0, invSigma, d+1) ! invSigma := X**T . X
    do i = 1,d
        invSigma(i,i) = invSigma(i,i) + lambda ! invSigma := invSigma + lambda*identity
    end do
    call dgemv('T', n, d+1, 1.d0, Xones, n, y_train, 1, 0.d0, w, 1) ! w := X**T . y_train
    call dgesv(d+1, 1, invSigma, d+1, ipiv, w, d+1, info) ! w := invSigma**-1 . w
end subroutine

subroutine ridge_regression_train_gradient_descent(n, d, X_train, y_train, lambda, learning_rate, n_iterations, w)
    implicit none
    integer, intent(in) :: n, d
    double precision, intent(in) :: X_train(n,d), y_train(n), lambda, learning_rate
    integer, intent(in) :: n_iterations
    double precision, intent(out) :: w(d+1)
    
    double precision :: Xones(n,d+1), y_err(n), delta_w(d+1), lambda_arr(d+1)
    integer :: i
    
    Xones(:,1:d) = X_train
    Xones(:,d+1) = 1.d0
    w = 0.d0
    lambda_arr = lambda
    lambda_arr(d+1) = 0.d0

    do i=1,n_iterations
        call dgemv('N', n, d+1, 1.d0, Xones, n, w, 1, 0.d0, y_err, 1) ! y_err := X . w
        call daxpy(n, -1.d0, y_train, 1, y_err, 1) ! y_err := y_err - y_train
        call dgemv('T', n, d+1, -learning_rate, Xones, n, y_err, 1, 0.d0, delta_w, 1) ! delta_w := -learning_rate * X**T . y_err
        w = w * (1.d0 - learning_rate*lambda_arr) + delta_w ! w := w - learning_rate * (X**T . y_err + lambda)
    end do
    
end subroutine

subroutine ridge_regression_predict(n_test, d, X_test, w, y_test)
    implicit none
    integer, intent(in) :: n_test, d
    double precision, intent(in) :: X_test(n_test, d)
    double precision, intent(in) :: w(d+1)
    double precision, intent(out) :: y_test(n_test)
    
    double precision :: Xones(n_test, d+1)
    
    Xones(:,1:d) = X_test
    Xones(:,d+1) = 1.d0
    
    call dgemv('N', n_test, d+1, 1.d0, Xones, n_test, w, 1, 0.d0, y_test, 1) ! y_test = X_test . w
end subroutine

subroutine ridge_regression_find_lambda(n, d, X_train, y_train, K, lambda_min, lambda_max, lambda_step, best_lambda)
    integer, intent(in) :: n, d
    double precision, intent(in) :: X_train(n, d), y_train(n)
    integer, intent(in) :: K
    double precision, intent(in) :: lambda_min, lambda_max, lambda_step
    double precision, intent(out) :: best_lambda

    double precision :: lambda, w(d+1), mse, min_mse
    integer :: nK, nstep
    integer :: ilambda, iK, ibin, iline, itrain
    double precision, allocatable :: X_train_cv(:,:), y_train_cv(:), X_test_cv(:,:), y_test_cv(:), &
                                     y_test_cv_ref(:), error(:)

    nK = n/K ! number of lines per training bin
    nnK = n - (K-1)*nK ! number of lines in the testing bin
    allocate(X_train_cv(nK*K,d))
    allocate(y_train_cv(nK*K))
    allocate(X_test_cv(nnK,d))
    allocate(y_test_cv_ref(nnK))
    allocate(y_test_cv(nnK))
    allocate(error(nnK))

    nstep = int((lambda_max - lambda_min)/lambda_step) + 1
    best_lambda = -1.d0
    min_mse = huge(min_mse)

    do ilambda=1,nstep
        lambda = lambda_min + lambda_step*(ilambda-1)
        mse = 0.d0
        do iK=1,K
            iline = 1
            itrain = 1
            do ibin=1,K
                if (ibin == iK) then ! testing bin
                    X_test_cv(:,:) = X_train(iline:iline+nnK-1,:)
                    y_test_cv_ref(:) = y_train(iline:iline+nnK-1)
                    iline = iline + nnK
                else ! training bin
                    X_train_cv(itrain:itrain+nK-1,:) = X_train(iline:iline+nK-1,:)
                    y_train_cv(itrain:itrain+nK-1) = y_train(iline:iline+nK-1)
                    itrain = itrain + nK
                    iline = iline + nK
                end if
            end do
            call ridge_regression_train(nK*(K-1), d, X_train_cv, y_train_cv, lambda, w)
            call ridge_regression_predict(nnK, d, X_test_cv, w, y_test_cv)
            error = y_test_cv - y_test_cv_ref
            mse = mse + dot_product(error, error)/dble(nnK)
        end do
        mse = mse / dble(K)
        !print *, "lambda = ", lambda, " mse = ", mse
        if (mse < min_mse) then
            min_mse = mse
            best_lambda = lambda
        end if
    end do
    !print *, "BEST LAMBDA = ", best_lambda, " mse = ", min_mse

    deallocate(X_train_cv, y_train_cv, X_test_cv, y_test_cv_ref, y_test_cv, error)
end subroutine