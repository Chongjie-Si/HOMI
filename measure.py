import numpy

def subset_accuracy(Y, P, O):
    n = (Y.shape[0] + P.shape[0]) // 2

    return numpy.sum(numpy.all((Y > 0.5) == (P > 0.5), 1)) / n


def hamming_loss(Y, P, O):
    n = (Y.shape[0] + P.shape[0]) // 2
    l = (Y.shape[1] + P.shape[1]) // 2

    s1 = numpy.sum(Y, 1)
    s2 = numpy.sum(P, 1)
    ss = numpy.sum(Y * P, 1)

    return numpy.sum(s1 + s2 - 2 * ss) / (n * l)


def accuracy(Y, P, O):
    n = (Y.shape[0] + P.shape[0]) // 2

    s1 = numpy.sum(Y, 1)
    s2 = numpy.sum(P, 1)
    ss = numpy.sum(Y * P, 1)
    sp = s1 + s2 - ss

    i = sp > 0

    return numpy.sum(ss[i] / sp[i]) / n


def precision(Y, P, O):
    n = (Y.shape[0] + P.shape[0]) // 2

    s1 = numpy.sum(Y, 1)
    s2 = numpy.sum(P, 1)
    ss = numpy.sum(Y * P, 1)

    i = s2 > 0

    return numpy.sum(ss[i] / s2[i]) / n


def recall(Y, P, O):
    n = (Y.shape[0] + P.shape[0]) // 2

    s1 = numpy.sum(Y, 1)
    s2 = numpy.sum(P, 1)
    ss = numpy.sum(Y * P, 1)

    i = s1 > 0

    return numpy.sum(ss[i] / s1[i]) / n


def f1(Y, P, O):
    p = precision(Y, P, O)
    r = recall(Y, P, O)
    return 2 * p * r / (p + r)


def macro_averaging_accuracy(Y, P, O):
    l = (Y.shape[1] + P.shape[1]) // 2

    tp = numpy.sum(Y * P, 0)
    tn = numpy.sum((1 - Y) * (1 - P), 0)
    fp = numpy.sum((1 - Y) * P, 0)
    fn = numpy.sum(Y * (1 - P), 0)

    ss = tp + tn
    sp = tp + fp + tn + fn

    i = sp > 0

    return numpy.sum(ss[i] / sp[i]) / l


def macro_averaging_precision(Y, P, O):
    l = (Y.shape[1] + P.shape[1]) // 2

    tp = numpy.sum(Y * P, 0)
    tn = numpy.sum((1 - Y) * (1 - P), 0)
    fp = numpy.sum((1 - Y) * P, 0)
    fn = numpy.sum(Y * (1 - P), 0)

    ss = tp
    sp = tp + fp

    i = sp > 0

    return numpy.sum(ss[i] / sp[i]) / l


def macro_averaging_recall(Y, P, O):
    l = (Y.shape[1] + P.shape[1]) // 2

    tp = numpy.sum(Y * P, 0)
    tn = numpy.sum((1 - Y) * (1 - P), 0)
    fp = numpy.sum((1 - Y) * P, 0)
    fn = numpy.sum(Y * (1 - P), 0)

    ss = tp
    sp = tp + fn

    i = sp > 0

    return numpy.sum(ss[i] / sp[i]) / l


def macro_averaging_f1(Y, P, O):
    l = (Y.shape[1] + P.shape[1]) // 2

    tp = numpy.sum(Y * P, 0)
    tn = numpy.sum((1 - Y) * (1 - P), 0)
    fp = numpy.sum((1 - Y) * P, 0)
    fn = numpy.sum(Y * (1 - P), 0)

    ss = 2 * tp
    sp = 2 * tp + fp + fn

    i = sp > 0

    return numpy.sum(ss[i] / sp[i]) / l


def micro_averaging_accuracy(Y, P, O):
    tp = numpy.sum(Y * P)
    tn = numpy.sum((1 - Y) * (1 - P))
    fp = numpy.sum((1 - Y) * P)
    fn = numpy.sum(Y * (1 - P))

    return (tp + tn) / (tp + fp + tn + fn)


def micro_averaging_precision(Y, P, O):
    tp = numpy.sum(Y * P)
    tn = numpy.sum((1 - Y) * (1 - P))
    fp = numpy.sum((1 - Y) * P)
    fn = numpy.sum(Y * (1 - P))

    return tp / (tp + fp)


def micro_averaging_recall(Y, P, O):
    tp = numpy.sum(Y * P)
    tn = numpy.sum((1 - Y) * (1 - P))
    fp = numpy.sum((1 - Y) * P)
    fn = numpy.sum(Y * (1 - P))

    return tp / (tp + fn)


def micro_averaging_f1(Y, P, O):
    tp = numpy.sum(Y * P)
    tn = numpy.sum((1 - Y) * (1 - P))
    fp = numpy.sum((1 - Y) * P)
    fn = numpy.sum(Y * (1 - P))

    return 2 * tp / (2 * tp + fp + fn)


def one_error(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2

    i = numpy.argmax(O, 1)

    return numpy.sum(1 - Y[range(n), i]) / n


def coverage(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2
    l = (Y.shape[1] + O.shape[1]) // 2

    R = numpy.array(O)
    i = numpy.argsort(O, 1)
    for r in range(n):
        R[r][i[r]] = range(l, 0, -1)

    return numpy.sum(numpy.max(R * Y, 1) - 1) / (n * l)


def ranking_loss(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2
    l = (Y.shape[1] + O.shape[1]) // 2

    p = numpy.zeros(n)
    q = numpy.sum(Y, 1)

    r, c = numpy.nonzero(Y)
    for i, j in zip(r, c): 
        p[i] += numpy.sum((Y[i, : ] < 0.5) * (O[i, : ] >= O[i, j]))

    i = (q > 0) * (q < l)

    return numpy.sum(p[i] / (q[i] * (l - q[i]))) / n


def average_precision(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2
    l = (Y.shape[1] + O.shape[1]) // 2

    R = numpy.array(O)
    i = numpy.argsort(O, 1)
    for r in range(n):
        R[r][i[r]] = range(l, 0, -1)

    p = numpy.zeros(n)
    q = numpy.sum(Y, 1)

    r, c = numpy.nonzero(Y)
    for i, j in zip(r, c):
        p[i] += numpy.sum((Y[i, : ] > 0.5) * (O[i, : ] >= O[i, j])) / R[i, j]

    i = q > 0

    return numpy.sum(p[i] / q[i]) / n


def macro_averaging_auc(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2
    l = (Y.shape[1] + O.shape[1]) // 2

    p = numpy.zeros(l)
    q = numpy.sum(Y, 0)

    r, c = numpy.nonzero(Y)
    for i, j in zip(r, c):
        p[j] += numpy.sum((Y[ : , j] < 0.5) * (O[ : , j] <= O[i, j]))

    i = (q > 0) * (q < n)

    return numpy.sum(p[i] / (q[i] * (n - q[i]))) / l


def micro_averaging_auc(Y, P, O):
    n = (Y.shape[0] + O.shape[0]) // 2
    l = (Y.shape[1] + O.shape[1]) // 2

    p = 0
    q = numpy.sum(Y)

    r, c = numpy.nonzero(Y)
    for i, j in zip(r, c):
        p += numpy.sum((Y < 0.5) * (O <= O[i, j]))

    return p / (q * (n * l - q))
