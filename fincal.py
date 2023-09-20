# Future Value
def fv(r, n, pv):
    return pv*(1+r)**n

# Present Value
def pv(r, n, fv):
    return fv/(1+r)**n

# PerpetuityPV
def pvPerp(c, r):
    return c/r

# Growing Perp
def pvGrowingPerp(c, r, g):
    return c/(r-g)

# PV Perp at T
def pvPerpAtT(c, r, t):
    return (c/r)*(1/(1+r)**(t-1))

# PV annuity
def pvAnnuity(c, r, n, due=0):
    if due: # if due = 1, then it's annuity due
        return (c/r)*(1-1/(1+r)**n)*(1+r)
    return (c/r)*(1-1/(1+r)**n)

# FV Annuity
def fvAnnuity(c, r, n, due=0):
    if due:
        return (c/r)*((1+r)**n-1)*(1+r)
    return (c/r)*((1+r)**n-1)

# Annuity Due
pvAnnuity(500, .04, 5, 1)
fvAnnuity(500, .04, 5, 1)

# PV Growing Annuity
def pvGrowingAnnuity(c, r, g, n):
    return (c/(r-g))*(1-((1+g)**n/(1+r)**n))

# FV Growing Annuity
def fvGrowingAnnuity(c, r, g, n):
    return (c/(r-g))*((1+r)**n - (1+g)**n)

# Monthly Payment
def pmt(r, n, pv):
    return (pv*r*(1+r)**n)/((1+r)**n-1)