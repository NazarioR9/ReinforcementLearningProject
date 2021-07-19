
def show_line_world(cells_count, v, line=0):
    for c in range(cells_count):
        print("|{:.7f}|".format(v[line * cells_count + c]), end='')
    print()


def show_grid_world(grid_count, v):
    for line in range(grid_count):
        show_line_world(grid_count, v, line)
