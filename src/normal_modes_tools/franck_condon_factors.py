import math

def franck_condon_factor(
        dq: float,
        nu_ini: float, nu_fin: float,
        n_ini: int, n_fin: int,
) -> float:
    """ A Franck-Condon factor for vibrational wavefunction along the same
    mode, displaced by `dq`. The mode is allowed to have a different frequency
    in both states `nu_ini` and `nu_fin`. The displacemet `dq` is
    expected in Å√amu. 

    Let's see:
        >>> franck_condon_factor(0.0, 1000, 1000, 0, 0)
        1.0
        >>> franck_condon_factor(0.0, 1000, 500, 0, 0)
        1.0
        # well ...

    """

    # Most popular units of 1930, at least according to E. Hutchinson
    aa_to_cm = 1e-8
    amu_to_gram: float = 1.6605655e-24
    sqrt_amu_to_sqrt_gram = math.sqrt(amu_to_gram)

    dq *= aa_to_cm * sqrt_amu_to_sqrt_gram

    c_cm_per_sec = 2.99792458e10  # still great units in use
    h_ERGxSEC = 6.626176e-27

    alpha = math.sqrt(nu_ini/nu_fin)
    delta = dq * 2 * math.pi 
    delta *= math.sqrt(c_cm_per_sec * nu_ini * amu_to_gram / h_ERGxSEC)

    fcf = 0.0
    for l in range(min(n_ini, n_fin) + 1):

        i_lim = 0
        if (n_fin - l) % 2 == 0:
            i_lim = (n_fin - l) // 2
        else:
            i_lim = (n_fin - l - 1) // 2
        i_lim += 1

        for i in range(i_lim):

            j_lim = 0
            if (n_ini - l) % 2 == 0:
                j_lim = (n_ini - l) // 2
            else:
                j_lim = (n_ini - l - 1) // 2
            j_lim +=1

            for j in range(j_lim):
                a2l =  1.0 / math.factorial(l)\
                    * math.pow(4.0 * alpha / (1 + alpha**2), l)

                b2i  = 1.0 / math.factorial(i)\
                    * math.pow((1 - alpha**2)/(1 + alpha**2), i)

                c2j  = 1.0 / math.factorial(j)\
                    * math.pow(-(1 - alpha**2)/(1 + alpha**2), j)

                dxx  = 1.0 / math.factorial(n_fin - 2*i - l)\
                * math.pow(
                    -2*alpha*delta / (1 + alpha**2),
                    n_fin - 2*i - l,
                )

                exx  = 1.0 / math.factorial(n_ini - 2*j - l)\
                        * math.pow(
                            2*alpha*delta / (1 + alpha **2),
                            n_ini - 2*j - l,
                        )

                prod = a2l * b2i * c2j * dxx * exx

                fcf += prod

    prefix  = \
            math.sqrt(math.factorial(n_ini)\
            * math.factorial(n_fin))\
            / math.pow(2 , (n_ini + n_fin) // 2)

    c3  = math.exp(-delta**2/ (2 * (1 + alpha**2)) )
    # a bit of divergence in ezFCF and Hutchinson
    # TODO: figure it out
    c3 *= math.sqrt(2*alpha/(1 + alpha**2))

    fcf *= prefix


    return fcf
