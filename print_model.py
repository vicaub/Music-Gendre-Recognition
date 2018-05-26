from keras.utils import plot_model
from keras.models import load_model

if __name__ == "__main__":
    import optparse
    optparser = optparse.OptionParser()
    optparser.add_option(
        '-m', '--model', type='string', default="cnn.h5",
        help='name of the model to use')

    options, args = optparser.parse_args()

    model = load_model('./models/' + options.model) 

    plot_model(model, to_file=options.model + ".png")