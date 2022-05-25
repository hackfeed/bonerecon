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

    x = ['100%', '75%', '50%', '25%']
    y1 = [17.5, 13.18, 11.48, 10.46]
    y2 = [14.4, 12.68, 11.03, 10.37]

    plt.figure(2)
    plt.bar(x, y1, color='red')
    plt.bar(x, y2, color='green')
    plt.title('Зависимость времени работы приложения от размера снимка')
    plt.xlabel('Размер снимка, % от исходного')
    plt.ylabel('Время работы приложения, сек.')
    plt.legend(['С классификацией', 'Без классификации'])
    plt.savefig('run.svg', format='svg')


if __name__ == '__main__':
    main()
