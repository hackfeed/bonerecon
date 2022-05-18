from tensorflow.keras.models import load_model


def main():
    model = load_model('../trained/teeth.h5')
    print(model.history)


if __name__ == '__main__':
    main()
