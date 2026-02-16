#!/usr/bin/python

"""A simple implementation of persistent homology on 2D images."""

__author__ = "Stefan Huber <shuber@sthu.org>"


from . import union_find


def get(im, p):
    return im[p[0]][p[1]]


def iter_neighbors(p, w, h):
    y, x = p

    # 8-neighborship
    neigh = [(y+j, x+i) for i in [-1, 0, 1] for j in [-1, 0, 1]]
    # 4-neighborship
    # neigh = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]

    for j, i in neigh:
        if j < 0 or j >= h:
            continue
        if i < 0 or i >= w:
            continue
        if j == y and i == x:
            continue
        yield j, i


def persistence(im):
    h, w = im.shape

    # Get indices orderd by value from high to low. As a tie-breaker, we use
    # (value, index) as key.
    indices = [(i, j) for i in range(h) for j in range(w)]
    # We add p as a secondary key to have an unambiguous total order below when
    # we enumerate neighboring cells of cells and consistency regarding
    # "oldest" component.
    indices.sort(key=lambda p: (get(im, p), p), reverse=True)

    # Maintains the growing sets
    uf = union_find.UnionFind()

    groups0 = {}

    def get_comp_birth(p):
        return get(im, uf[p])

    # Process pixels from high to low
    for i, p in enumerate(indices):
        v = get(im, p)
        ni = [uf[q] for q in iter_neighbors(p, w, h) if q in uf]
        # Sort by (value, index) as key. Note that this is the same sorting
        # order as for indices. Otherwise we have an inconsistence notion of
        # the "older" component!
        nc = sorted([(get_comp_birth(q), q) for q in set(ni)], reverse=True)

        if i == 0:
            groups0[p] = (v, v, None)

        uf.add(p, -i)

        if len(nc) > 0:
            oldp = nc[0][1]
            uf.union(oldp, p)

            # Merge all others with oldp
            for bl, q in nc[1:]:
                if uf[q] not in groups0:
                    # print(i, ": Merge", uf[q], "with", oldp, "via", p)
                    groups0[uf[q]] = (bl, bl-v, p)
                uf.union(oldp, q)

    groups0 = [(k, groups0[k][0], groups0[k][1], groups0[k][2]) for k in groups0]
    groups0.sort(key=lambda g: g[2], reverse=True)

    return groups0
