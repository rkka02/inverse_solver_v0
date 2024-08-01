import numpy as np

def phi_shift(restored_p2, n=1):
    imY, imX = restored_p2.shape
    bsize = 8

    mask = np.ones((imY, imX))
    mask[bsize:imY-bsize, bsize:imX-bsize] = 0

    imY, imX = restored_p2.shape
    XX, YY = np.meshgrid(np.arange(imX), np.arange(imY))
    p2mask = restored_p2 * mask
    
    # Omit Outliers
    list_ = p2mask[p2mask != 0].flatten()
    p25 = np.percentile(list_, 25)
    p75 = np.percentile(list_, 75)
    cmin = p25 - 1.5 * (p75 - p25)
    cmax = p75 + 1.5 * (p75 - p25)
    p2mask[p2mask < cmin] = 0
    p2mask[p2mask > cmax] = 0
    
    # Find Coeffitients(?)
    p2mask = p2mask.flatten()
    X = np.zeros((len(p2mask[p2mask!=0]), n))
    Y = np.zeros((len(p2mask[p2mask!=0]), n))
    for ii in range(n):
        XXX = (XX**1).flatten()
        YYY = (YY**1).flatten()
        XXX = XXX[p2mask!=0]
        YYY = YYY[p2mask!=0]
        X[:, ii] = XXX
        Y[:, ii] = YYY
        
    p2mask = p2mask[p2mask!=0]
    E = np.ones(len(p2mask))
    AA = np.column_stack((X, Y, E))
    coefficients = np.linalg.lstsq(AA, p2mask, rcond=None)[0]
    
    goodp2 = restored_p2 - coefficients[-1] * np.ones((imY, imX))
    for ii in range(n):
        goodp2 -= coefficients[ii] * XX**(ii+1)
        goodp2 -= coefficients[n + ii] * YY**(ii+1)
    
    return goodp2