from scipy.stats import norm
import numpy as np
from scipy.special import erfc, erfcx

def int_1_indep(a,b,c,d):
    #normalization
    return 1/np.sqrt(a+1)*np.exp((b**2+d**2)/(2*(a+1)))*np.cosh(c+b*d/(a+1))

def int_2_indep(a,b,c,d):
    #over lambda
    a = a+1
    return 1/(np.sqrt(a)*a)*np.exp((b**2+d**2)/(2*a))*(b*np.cosh(c+b*d/a) + d*np.sinh(c+b*d/a))

def int_3_indep(a,b,c,d):
    #over nu
    a = a+1
    return 1/np.sqrt(a)*np.exp((b**2+d**2)/(2*a))*np.sinh(c+b*d/a)

def int_4_indep(a,b,c,d):
    #over lambda and nu
    a = a+1
    return 1/(np.sqrt(a)*a)*np.exp((b**2+d**2)/(2*a))*(d*np.cosh(c+b*d/a) + b*np.sinh(c+b*d/a))

def int_5_indep(a,b,c,d):
    #over lambda^2
    a = a+1
    return 1/(np.sqrt(a)*a)*np.exp((b**2+d**2)/(2*a))*((1+(b**2+d**2)/a)*np.cosh(c+b*d/a) + 2*b*d/a*np.sinh(c+b*d/a))

#Before changing to erfcx

def int_1_corr(a,b,c,d):
    #normalization
    a = a+1
    return 1/np.sqrt(a)*(np.exp(-c)*np.exp((d-b)**2/(2*a))*erfc(-(d-b)/np.sqrt(2*a))/2 + np.exp(c)*np.exp((b+d)**2/(2*a))*erfc(-(b+d)/np.sqrt(2*a))/2)

def int_2_corr(a,b,c,d):
    #over lambda
    a = a+1
    return 1/(np.sqrt(a)*a)*(np.exp(c)*np.exp(((b+d)**2)/(2*a))*(d+b)*erfc(-(d+b)/np.sqrt(2*a))/2-np.exp(-c)*np.exp((d-b)**2/(2*a))*erfc(-(d-b)/np.sqrt(2*a))/2*(d-b)) + 1/(np.sqrt(2*np.pi)*a)*(np.exp(c)-np.exp(-c))

def int_3_corr(a,b,c,d):
    #over nu
    a = a+1
    return 1/np.sqrt(a)*(np.exp(c)*np.exp((b+d)**2/(2*a))*erfc(-(b+d)/np.sqrt(2*a))/2-np.exp(-c)*np.exp((d-b)**2/(2*a))*erfc(-(d-b)/np.sqrt(2*a))/2) 

def int_4_corr(a,b,c,d):
    # over lambda and nu
    a = a+1
    return 1/(np.sqrt(a)*a)*(np.exp(c)*np.exp((b+d)**2/(2*a))*(b+d)*erfc(-(b+d)/np.sqrt(2*a))/2 + np.exp(-c)*np.exp((d-b)**2/(2*a))*(d-b)*erfc(-(d-b)/np.sqrt(2*a))/2) + 1/(np.sqrt(2*np.pi)*a)*(np.exp(c)+np.exp(-c))

def int_5_corr(a,b,c,d):
    # over lambda^2
    a = a+1
    return 1/(np.sqrt(a)*a)*((1+(b+d)**2/a)*np.exp(c)*np.exp((b+d)**2/(2*a))*erfc(-(b+d)/np.sqrt(2*a))/2 + (1+(d-b)**2/a)*np.exp(-c)*np.exp((d-b)**2/(2*a))*erfc(-(d-b)/np.sqrt(2*a))/2) + np.sqrt(2/np.pi)/a**2*(b*np.sinh(c)+d*np.cosh(c))


def gamma_matrix(beta_v):
    mat = np.zeros((2,2))
    mat[1,1] = beta_v
    return mat

def int_1_dep(a,b,c,d):
    # normalization
    a = a+1
    t = norm.ppf(0.75)
    return 1/np.sqrt(a)*(np.exp(-c)*np.exp((d-b)**2/(2*a))*(erfc((-t-(b-d)/a)*np.sqrt(a/2))/2-erfc((t-(b-d)/a)*np.sqrt(a/2))/2) + np.exp(c)*np.exp((b+d)**2/(2*a))*(erfc((t+(b+d)/a)*np.sqrt(a/2))/2+erfc((t-(b+d)/a)*np.sqrt(a/2))/2))

def int_2_dep(a,b,c,d):
    # over lambda
    a = a+1
    t = norm.ppf(0.75)
    first_piece = -np.exp(c)*np.exp((b+d)**2/(2*a))*(-(b+d)/a*erfc((t+(b+d)/a)*np.sqrt(a/2))/2+1/np.sqrt(2*np.pi*a)*np.exp(-(t+(b+d)/a)**2/2*a))
    second_piece = np.exp(-c)*np.exp((d-b)**2/(2*a))*((b-d)/a*(erfc(np.sqrt(a/2)*(-t-(b-d)/a))/2-erfc(np.sqrt(a/2)*(t-(b-d)/a))/2)+1/np.sqrt(2*np.pi*a)*(np.exp(-(-t-(b-d)/a)**2/2*a)-np.exp(-(t-(b-d)/a)**2/2*a)))
    third_piece = np.exp(c)*np.exp((b+d)**2/(2*a))*((b+d)/a*erfc(np.sqrt(a/2)*(t-(b+d)/a))/2+1/np.sqrt(2*np.pi*a)*np.exp(-(t-(b+d)/a)**2/2*a))
    return 1/np.sqrt(a)*(first_piece + second_piece + third_piece)

def int_3_dep(a,b,c,d):
    # over nu
    a = a+1
    t = norm.ppf(0.75)
    return 1/np.sqrt(a)*(-np.exp(-c)*np.exp((d-b)**2/(2*a))*(erfc((-t-(b-d)/a)*np.sqrt(a/2))/2-erfc((t-(b-d)/a)*np.sqrt(a/2))/2) + np.exp(c)*np.exp((b+d)**2/(2*a))*(erfc((t+(b+d)/a)*np.sqrt(a/2))/2+erfc((t-(b+d)/a)*np.sqrt(a/2))/2))

def int_4_dep(a,b,c,d):
    # over lambda and nu
    a = a+1
    t = norm.ppf(0.75)
    first_piece = -np.exp(c)*np.exp((b+d)**2/(2*a))*(-(b+d)/a*erfc((t+(b+d)/a)*np.sqrt(a/2))/2+1/np.sqrt(2*np.pi*a)*np.exp(-(t+(b+d)/a)**2/2*a))
    second_piece = -np.exp(-c)*np.exp((d-b)**2/(2*a))*((b-d)/a*(erfc(np.sqrt(a/2)*(-t-(b-d)/a))/2-erfc(np.sqrt(a/2)*(t-(b-d)/a))/2)+1/np.sqrt(2*np.pi*a)*(np.exp(-(-t-(b-d)/a)**2/2*a)-np.exp(-(t-(b-d)/a)**2/2*a)))
    third_piece = np.exp(c)*np.exp((b+d)**2/(2*a))*((b+d)/a*erfc(np.sqrt(a/2)*(t-(b+d)/a))/2+1/np.sqrt(2*np.pi*a)*np.exp(-(t-(b+d)/a)**2/2*a))
    return 1/np.sqrt(a)*(first_piece + second_piece + third_piece)

def int_5_dep(a,b,c,d):
    # over lambda^2
    a = a+1
    t = norm.ppf(0.75)
    first_piece = 1/(a*np.sqrt(a)) * np.exp(c) * np.exp((b+d)**2/(2*a)) * (1+(b+d)**2/a) * (erfc(np.sqrt(a/2)*(t+(b+d)/a))/2 + erfc(np.sqrt(a/2)*(t-(b+d)/a))/2)
    second_piece = np.sqrt(2/np.pi)/a * np.exp(c) * np.exp(-t**2*a/2) * (t*np.cosh(t*(b+d)) + (b+d)/a*np.sinh(t*(b+d)))
    third_piece = -1/(np.sqrt(2*np.pi)*a)*np.exp(-c)*np.exp(-t**2*a/2)*(2*t*np.cosh(t*(b-d)) + 2*(b-d)/a*np.sinh(t*(b-d)))
    fourth_piece = 1/(np.sqrt(a)*a)*np.exp(-c)*np.exp((b-d)**2/(2*a))*(1+(b-d)**2/a)*(erfc(np.sqrt(a/2)*(-t-(b-d)/a))/2-erfc(np.sqrt(a/2)*(t-(b-d)/a))/2)
    return first_piece + second_piece + third_piece + fourth_piece


def int_1(a,b,c,d, gamma, delta):
    if gamma == 1:
        return int_1_corr(a, b, c, d)
    elif delta == 1:
        return int_1_dep(a, b, c, d)
    elif gamma == 0 and delta == 0:
        return int_1_indep(a, b, c, d)
    else:
        return gamma*int_1_corr(a, b, c, d) + delta*int_1_dep(a, b, c, d) + (1-gamma-delta)*int_1_indep(a, b, c, d)

def int_2(a,b,c,d, gamma, delta):
    if gamma == 1:
        return int_2_corr(a, b, c, d)
    elif delta == 1:
        return int_2_dep(a, b, c, d)
    elif gamma == 0 and delta == 0:
        return int_2_indep(a, b, c, d)
    else:   
        return gamma*int_2_corr(a, b, c, d) + delta*int_2_dep(a, b, c, d) + (1-gamma-delta)*int_2_indep(a, b, c, d)

def int_3(a,b,c,d, gamma, delta):
    if gamma == 1:
        return int_3_corr(a, b, c, d)
    elif delta == 1:
        return int_3_dep(a, b, c, d)
    elif gamma == 0 and delta == 0:
        return int_3_indep(a, b, c, d)
    else:
        return gamma*int_3_corr(a, b, c, d) + delta*int_3_dep(a, b, c, d) + (1-gamma-delta)*int_3_indep(a, b, c, d)

def int_4(a,b,c,d, gamma, delta):
    if gamma == 1:
        return int_4_corr(a, b, c, d)
    elif delta == 1:
        return int_4_dep(a, b, c, d)
    elif gamma == 0 and delta == 0:
        return int_4_indep(a, b, c, d)
    else:
        return gamma*int_4_corr(a, b, c, d) + delta*int_4_dep(a, b, c, d) + (1-gamma-delta)*int_4_indep(a, b, c, d)

def int_5(a,b,c,d, gamma, delta):
    if gamma == 1:
        return int_5_corr(a, b, c, d)
    elif delta == 1:
        return int_5_dep(a, b, c, d)
    elif gamma == 0 and delta == 0:
        return int_5_indep(a, b, c, d)
    else:
        return gamma*int_5_corr(a, b, c, d) + delta*int_5_dep(a, b, c, d) + (1-gamma-delta)*int_5_indep(a, b, c, d)
    
def corr_int(gamma):
    return gamma*np.sqrt(2/np.pi)

def I1I2I3_gauss_hermite(a, b, c, d, e, n=200):
    """
    Compute I1, I2, I3 for:
      lambda ~ N(0,1), s in {-1,+1} equiprob, nu = s*|lambda|
    using Gauss-Hermite quadrature of order n.
    Returns (I1, I2, I3).
    """
    # Hermite nodes/weights for ∫ exp(-x^2) f(x) dx
    x, w = np.polynomial.hermite.hermgauss(n)

    # Convert to standard normal: Z = sqrt(2) * x
    z = np.sqrt(2.0) * x
    absz = np.abs(z)
    zabsz = z * absz

    # Common factor inside expectation
    expo = np.exp(-0.5 * (a + e) * z**2 + b * z)

    # cosh/sinh argument after averaging over s
    arg = c * absz + d * zabsz

    f1 = expo * np.cosh(arg)
    f2 = z * expo * np.cosh(arg)
    f3 = absz * expo * np.sinh(arg)

    # E[f(Z)] = (1/sqrt(pi)) * Σ w_i f( sqrt(2) x_i )
    pref = 1.0 / np.sqrt(np.pi)

    I1 = pref * np.sum(w * f1)
    I2 = pref * np.sum(w * f2)
    I3 = pref * np.sum(w * f3)

    return I1, I2, I3
