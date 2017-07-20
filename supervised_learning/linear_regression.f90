subroutine linear_regression_train(n, d, X_train, y_train, w)
    implicit none
    integer, intent(in) :: n, d
    double precision, intent(in) :: X_train(n,d), y_train(n)
    double precision, intent(out) :: w(d+1)
    
    integer :: info, lwork
    double precision :: Xones(n,d+1), rhs(n)
    double precision, allocatable :: work(:)
    
    Xones(:,1:d) = X_train
    Xones(:,d+1) = 1.d0
    rhs(:) = y_train
    
    lwork = -1
    info = 0
    allocate(work(1))
    call dgels('N', n, d+1, 1, Xones, n, rhs, n, work, lwork, info)
    lwork = int(work(1))
    deallocate(work)
    allocate(work(lwork))
    call dgels('N', n, d+1, 1, Xones, n, rhs, n, work, lwork, info)
    deallocate(work)
    
    w = rhs(1:d+1)
end subroutine

subroutine linear_regression_train_gradient_descent(n, d, X_train, y_train, learning_rate, n_iterations, w)
    implicit none
    integer, intent(in) :: n, d
    double precision, intent(in) :: X_train(n,d), y_train(n), learning_rate
    integer, intent(in) :: n_iterations
    double precision, intent(out) :: w(d+1)
    
    double precision :: Xones(n,d+1), y_err(n), delta_w(d+1)
    integer :: i
    
    Xones(:,1:d) = X_train
    Xones(:,d+1) = 1.d0
    w = 0.d0
    do i=1,n_iterations
        call dgemv('N', n, d+1, 1.d0, Xones, n, w, 1, 0.d0, y_err, 1) ! y_err := X . w
        call daxpy(n, -1.d0, y_train, 1, y_err, 1) ! y_err := y_err - y_train
        call dgemv('T', n, d+1, -learning_rate, Xones, n, y_err, 1, 0.d0, delta_w, 1) ! delta_w := -learning_rate * X**T . y_err
        w = w + delta_w
    end do
    
end subroutine

subroutine linear_regression_predict(n_test, d, X_test, w, y_test)
    implicit none
    integer, intent(in) :: n_test, d
    double precision, intent(in) :: X_test(n_test, d)
    double precision, intent(in) :: w(d+1)
    double precision, intent(out) :: y_test(n_test)
    
    double precision :: Xones(n_test, d+1)
    
    Xones(:,1:d) = X_test
    Xones(:,d+1) = 1.d0
    
    call dgemv('N', n_test, d+1, 1.d0, Xones, n_test, w, 1, 0.d0, y_test, 1)
end subroutine
