def judgeifadd(minx1, miny1, maxx1, maxy1, minx2, miny2, maxx2, maxy2):
    minx = max(minx1, minx2)
    miny = max(miny1, miny2)
    maxx = min(maxx1, maxx2)
    maxy = min(maxy1, maxy2)
    if minx < maxx and miny < maxy:
        return True
    else:
        return False

print(judgeifadd(1497,  122, 1630,  456, 774,  815,  904, 1080))