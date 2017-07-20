subroutine gaussian_naive_bayes_train(n, d, K, X_train, y_train, pi, mu, Sigma)
    implicit none
    integer, intent(in) :: n, d, K
    double precision, intent(in) :: X_train(n,d)
    integer, intent(in) :: y_train(n)
    double precision, intent(out) :: pi(K), mu(K,d), Sigma(K,d,d)

    type array
        double precision, allocatable :: v(:,:)
    end type array

    integer :: i, n_y(K), y, idelta(K)
    type(array) :: delta(K)

    pi = 0.d0
    mu = 0.d0
    Sigma = 0.d0
    n_y = 0

    do i=1,n
        y = y_train(i) + 1
        n_y(y) = n_y(y) + 1
        mu(y,:) = mu(y,:) + X_train(i,:)
    end do
    do y=1,K
        pi(y) = dble(n_y(y))/dble(n)
        mu(y,:) = mu(y,:)/n_y(y)
        allocate(delta(y)%v(n_y(y),d))
    end do
    idelta = 1
    do i=1,n
        y = y_train(i) + 1
        delta(y)%v(idelta(y),:) = X_train(i,:) - mu(y,:)
        idelta(y) = idelta(y) + 1
    end do
    print *, n_y
    print *, idelta
    do y=1,K
        call dgemm('T', 'N', d, d, n_y(y), 1.d0, delta(y)%v, n_y(y), delta(y)%v, n_y(y), 0.d0, Sigma(y,:,:), d) ! Sigma := delta**T . delta
        Sigma(y,:,:) = Sigma(y,:,:) / n_y(y)
        if (allocated(delta(y)%v)) then
            deallocate(delta(y)%v)
        end if
    end do
end subroutine

function multivariate_normal(d, x, mu, Sigma) result(f)
    implicit none
    integer, intent(in) :: d
    double precision, intent(in) :: x(d), mu(d), Sigma(d,d)
    double precision :: f

    double precision :: delta(d), Sigma_s(d*(d+1)/2), invSigma_delta(d), det
    double precision, parameter :: pi = 4.d0 * atan(1.d0)
    integer :: ipiv(d), info, i, j

    do i=1,d
        do j=i,d
            Sigma_s(i+(j-1)*j/2) = Sigma(i,j)
        end do
    end do

    delta = x-mu
    invSigma_delta = delta
    call dspsv("U", d, 1, Sigma, ipiv, invSigma_delta, d, info) ! invSigma_delta := Sigma**-1 . delta
    det = 1.d0
    do i=1,d
        if (ipiv(i) > 0) then
            det = det * Sigma_s(i+(i-1)*i/2)
        elseif (i>1 .and. ipiv(i) < 0 .and. ipiv(i-1) == ipiv(i)) then
            det = det * (Sigma_s(i+(i-1)*i/2) * Sigma_s(i-1+(i-1)*(i-2)/2) - &
                         Sigma_s(i-1+i*(i-1)/2) * Sigma_s(i+(i-1)*(i-2)/2))
        end if
    end do
    det = det * (2.d0*pi)**d
    f = exp(-.5d0 * dot_product(delta, invSigma_delta)) / det
end function

subroutine gaussian_naive_bayes_predict(n_test, d, K, X_test, pi, mu, sigma, y_test)
    implicit none
    integer, intent(in) :: n_test, d, K
    double precision, intent(in) :: X_test(n_test, d)
    double precision, intent(in) :: pi(K), mu(K,d), sigma(K,d,d)
    integer, intent(out) :: y_test(n_test)

    integer :: i, y, best_y
    double precision :: x(d), prob, best_prob
    double precision :: multivariate_normal
    
    do i=1,n_test
        x = X_test(i,:)
        best_prob = 0.d0
        do y=1,K
            prob = multivariate_normal(d, x, mu(y,:), Sigma(y,:,:)) * pi(y)
            if (prob >= best_prob) then
                best_prob = prob
                best_y = y
            end if
        end do
        y_test(i) = best_y
    end do
end subroutine
