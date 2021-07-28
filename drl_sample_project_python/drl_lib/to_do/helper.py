import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

PLOT_PATH = 'drl_lib/data/plots/'
SLIDE_PATH = 'drl_lib/data/slides/'
WEIGHT_PATH = 'drl_lib/data/weights/'


class Tracker:
    inner_name = 'tracker_'

    def __init__(self, name=None):
        self.elements = []
        self.name = name

    def __call__(self, r):
        self.elements.append(r)

    def overall(self):
        raise NotImplementedError()

    def format_path(self):
        return os.path.join(PLOT_PATH, self.inner_name + self.name)

    def save_to_file(self):
        assert self.name is not None

        save_to_pickle(self.elements, self.format_path())

    def load_from_file(self):
        assert self.name is not None

        self.elements = load_from_pickle(self.format_path())

    def save_to_image(self, x=None):
        name = self.format_path()

        if x is not None:
            plt.plot(x, self.elements)
        else:
            plt.plot(self.elements)
        plt.savefig(name + '.png')
        plt.show()


class RewardTracker(Tracker):
    inner_name = 'reward_'

    def __init__(self, name, stat_step=10, verbose=True):
        super().__init__(name)
        self.running = []
        self.steps = [0]
        self.running_step = 0
        self.stat_step = stat_step
        self.verbose = verbose
        self.locked = False

    def __call__(self, r):
        if self.locked:
            return

        self.running_step += 1
        self.running.append(r)

        if self.running_step % self.stat_step == 0:
            step_cum = np.mean(self.running)
            super().__call__(step_cum)
            self.steps.append(self.running_step)
            self.running.clear()

            if self.verbose:
                print(f'Mean reward at episode {self.running_step} is : {step_cum} - Overall is : {self.overall()}')

    def overall(self):
        if len(self.elements) == 0:
            return 0

        return np.mean(self.elements)

    def save_to_image(self, x=None):
        if not self.locked:
            self.elements = [0] + self.elements
            self.locked = True
        super(RewardTracker, self).save_to_image(self.steps)


class WinTracker(Tracker):
    inner_name = 'win_'

    def __count(self, i):
        return len(np.where(self.elements == i))

    def overall(self):
        if len(self.elements) == 0:
            return {}

        return {"Win": self.__count(1), "Draw": self.__count(0), "Lose": self.__count(-1)}


class TrainingPlotter(Tracker):
    def __init__(self, name):
        super().__init__(name)
        self.name = name

    def overall(self):
        return self.elements


def show_line_world(cells_count, v, line=0):
    for c in range(cells_count):
        print("|{:.7f}|".format(v[line * cells_count + c]), end='')
    print()


def show_grid_world(grid_count, v):
    for line in range(grid_count):
        show_line_world(grid_count, v, line)


def save_to_pickle(obj, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_from_pickle(path):
    with open(path + '.pkl', 'rb') as f:
        obj = pickle.load(f)

    return obj
