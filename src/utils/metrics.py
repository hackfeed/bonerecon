import pickle
import matplotlib.pyplot as plt


def main():
    with open('../train/trained/teeth.hist', 'rb') as f:
        data = pickle.load(f)

    plt.rcParams['font.family'] = 'Times New Roman'

    plt.figure(0)
    plt.plot(data['accuracy'])
    plt.title('Точность модели')
    plt.ylabel('Точность')
    plt.xlabel('Эпоха')
    plt.savefig('accuracy.svg', format='svg')

    plt.figure(1)
    plt.plot(data['loss'])
    plt.title('Фунцкция потерь')
    plt.ylabel('Потери')
    plt.xlabel('Эпоха')
    plt.savefig('loss.svg', format='svg')


if __name__ == '__main__':
    main()
