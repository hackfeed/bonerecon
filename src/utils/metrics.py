import pickle

import matplotlib.pyplot as plt
import numpy as np


def main():
    with open('../train/trained/teeth.hist', 'rb') as f:
        data = pickle.load(f)

    plt.rcParams['font.family'] = 'Times New Roman'

    plt.figure(0)
    plt.plot(data['accuracy'], color='green')
    plt.title(f"Точность модели (максимальное значение: {max(data['accuracy']):.3f})")
    plt.ylabel('Точность')
    plt.xlabel('Эпоха')
    plt.savefig('accuracy.svg', format='svg')

    plt.figure(1)
    plt.plot(data['loss'], color='red')
    plt.title(f"Фунцкция потерь (минимальное значение: {min(data['loss']):.3f})")
    plt.ylabel('Потери')
    plt.xlabel('Эпоха')
    plt.savefig('loss.svg', format='svg')


if __name__ == '__main__':
    main()
