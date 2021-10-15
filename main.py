
from Neuron import *
import matplotlib.pyplot as plt

# Training data
DX = [[0, 0], [0, 1], [1, 0], [1, 1]]
DX_bi = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
Dy = [0, 0, 0, 1]
Dy_bi = [-1, -1, -1, 1]


# Experiments
def exp11():
    list_ = []
    thetas = [0.01, 0.1, 0.5, 1, 2, 5]
    for theta in thetas:
        total = 0
        for i in range(repeats):
            neuron = Perceptron(eta=0.1, W_range=(0.01, -0.01), theta_range=(theta, theta), unipolar=False)
            neuron.fit(DX_bi, Dy_bi)
            total += neuron.eras
        list_.append(total / repeats)
    return thetas, list_


def exp12():
    list_ = []
    W_ranges = [(1, -1), (0.8, -0.8), (0.5, -0.5), (0.2, -0.2), (0.05, -0.05)]
    for range_ in W_ranges:
        total = 0
        for i in range(repeats):
            neuron = Perceptron(eta=0.1, W_range=range_, theta_range=(1, -1), unipolar=False)
            neuron.fit(DX_bi, Dy_bi)
            total += neuron.eras
        list_.append(total / repeats)
    return list(map(lambda x: x[0], W_ranges)), list_


def exp13():
    list_ = []
    etas = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5]
    for eta in etas:
        total = 0
        for i in range(repeats):
            neuron = Perceptron(eta=eta, W_range=(0.01, -0.01), theta_range=(1, -1), unipolar=False)
            neuron.fit(DX_bi, Dy_bi)
            total += neuron.eras
        list_.append(total / repeats)
    return etas, list_


def exp14():
    list_ = []
    types = ["unipolar", "bipolar"]
    for type_ in types:
        total = 0
        for i in range(repeats):
            neuron = Perceptron(eta=0.1, W_range=(0.01, -0.01), theta_range=(1, -1), unipolar=type_ == "unipolar")
            neuron.fit(DX_bi if type_ == "bipolar" else DX, Dy_bi if type_ == "bipolar" else Dy)
            total += neuron.eras
        list_.append(total / repeats)
    return types, list_


def exp21():
    list_ = []
    W_ranges = [(1, -1), (0.8, -0.8), (0.5, -0.5), (0.2, -0.2), (0.05, -0.05)]
    for range_ in W_ranges:
        total = 0
        for i in range(repeats):
            neuron = Adaline(eta=0.1, threshold=0.4, W_range=range_, theta_range=(1, -1))
            neuron.fit(DX_bi, Dy_bi)
            total += neuron.eras
        list_.append(total / repeats)
    return list(map(lambda x: x[0], W_ranges)), list_


def exp22():
    list_ = []
    etas = [0.001, 0.01, 0.1, 1]
    for eta in etas:
        total = 0
        for i in range(repeats):
            neuron = Adaline(eta=eta, threshold=0.4, W_range=(0.01, -0.01), theta_range=(1, -1))
            neuron.fit(DX_bi, Dy_bi)
            total += neuron.eras
        list_.append(total / repeats)
    return etas, list_


def exp23():
    list_ = []
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for threshold in thresholds:
        total = 0
        for i in range(repeats):
            neuron = Adaline(eta=0.1, threshold=threshold, W_range=(0.01, -0.01), theta_range=(1, -1))
            neuron.fit(DX_bi, Dy_bi)
            total += neuron.eras
        list_.append(total / repeats)
    return thresholds, list_


def exp24():
    list_ = []
    result = ["Perceptron", "Adaline"]
    total = 0
    for i in range(repeats):
        neuron = Perceptron(eta=0.2, W_range=(0.1, -0.1), theta_range=(0.5, -0.5), unipolar=False)
        neuron.fit(DX_bi, Dy_bi)
        total += neuron.eras
    list_.append(total / repeats)
    total = 0
    for i in range(repeats):
        neuron = Adaline(eta=0.1, threshold=0.6, W_range=(0.2, -0.2), theta_range=(0.5, -0.5))
        neuron.fit(DX_bi, Dy_bi)
        total += neuron.eras
    list_.append(total / repeats)
    return result, list_


def print_result(result_x, result_y, divider):
    print(f" {divider} ".join(map(str, result_x)))
    print(f" {divider} ".join(map(str, result_y)))


def make_exp(exp, xscale, title, x, y, dest, show, bar):
    print(f"Rozpoczeto {title.lower()}")
    exp_result = exp()
    print_result(*exp_result, "&")

    plt.bar(*exp_result) if bar else plt.plot(*exp_result)
    if xscale:
        plt.xscale(xscale)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show() if show else plt.savefig(dest)
    plt.clf()
    print(f"Zakonczono {title.lower()}\n")


show_tests = False
repeats = 100
#make_exp(exp=exp11, xscale="linear", title="Wpływ theta na szybkość uczenia (Perceptron)", x="próg", y="epoki", dest="plots/per_exp1.png", show=show_tests, bar=False)
#make_exp(exp=exp12, xscale="linear", title="Wpływ zakresu wag W na szybkość uczenia (Perceptron)", x="zakres", y="epoki", dest="plots/per_exp2.png", show=show_tests, bar=False)
#make_exp(exp=exp13, xscale="linear", title="Wpływ wspólczynnika uczenia na szybkość uczenia (Perceptron)", x="apha", y="epoki", dest="plots/per_exp3.png", show=show_tests, bar=False)
#make_exp(exp=exp14, xscale=None, title="Wspływ logiki na szybkość uczenia (Perceptron)", x="", y="epoki", dest="plots/per_exp4.png", show=show_tests, bar=True)

#make_exp(exp=exp21, xscale="linear", title="Wpływ zakresu wag W na szybkość uczenia (Adaline)", x="zakres", y="epoki", dest="plots/ada_exp1.png", show=show_tests, bar=False)
#make_exp(exp=exp22, xscale="log", title="Wpływ wspólczynnika uczenia na szybkość uczenia (Adaline)", x="alpha", y="epoki", dest="plots/ada_exp2.png", show=show_tests, bar=False)
#make_exp(exp=exp23, xscale="linear", title="Wpływ progu błędu na szybkość uczenia", x="próg (Adaline)", y="epoki", dest="plots/ada_exp3.png", show=show_tests, bar=False)
make_exp(exp=exp24, xscale="linear", title="Wpływ modelu na szybkość uczenia", x="model", y="epoki", dest="plots/ada_exp4.png", show=show_tests, bar=True)
