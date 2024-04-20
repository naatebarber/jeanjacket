from radial_dichotomy.dichotomy import RadialDichotomy

if __name__ == "__main__":
    vals = [110, 9, 12, 10]

    rd = RadialDichotomy(vals, 2, 5)

    rd.permute(False)

    for ix, lay in enumerate(rd.concentric):
        print(ix, len(lay.symbols))
