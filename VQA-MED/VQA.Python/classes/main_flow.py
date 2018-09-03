import itertools

from classes.vqa_model_builder import VqaModelBuilder
from classes.vqa_model_predictor import VqaModelPredictor
from classes.vqa_model_trainer import VqaModelTrainer
from keras import backend as keras_backend
from vqa_logger import logger
from common.utils import VerboseTimer
from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase


def main():
    # Create------------------------------------------------------------------------
    # good for a model to predict multiple mutually-exclusive classes:

    loss, activation = 'categorical_crossentropy', 'softmax'

    # loss, activation = 'binary_crossentropy', 'sigmoid'
    # loss, activation = 'categorical_crossentropy', 'sigmoid'

    losses = ['categorical_crossentropy', 'binary_crossentropy', 'kullback_leibler_divergence', 'poisson',
              'cosine_proximity', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
              'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh']
    activations = ['softmax', 'sigmoid', 'relu', 'tanh']

    losses_and_activations = list(itertools.product(losses, activations))

    for loss, activation in losses_and_activations:
        try:

            mb = VqaModelBuilder(loss, activation)
            model = mb.get_vqa_model()
            model_fn, summary_fn, fn_image = VqaModelBuilder.save_model(model)

            # Train ------------------------------------------------------------------------

            # epochs=25
            # batch_size = 20
            keras_backend.clear_session()
            epochs = 1
            batch_size = 75

            model_location = model_fn
            mt = VqaModelTrainer(model_location, epochs=epochs, batch_size=batch_size)
            history = mt.train()
            with VerboseTimer("Saving trained Model"):
                model_fn, summary_fn, fn_image, fn_history = VqaModelTrainer.save(mt.model, history)
            print(model_fn)

            # Evaluate ------------------------------------------------------------------------
            mp = VqaModelPredictor(model=None)
            validation_prediction = mp.predict(mp.df_validation)
            predictions = validation_prediction.prediction.values
            ground_truth = validation_prediction.answer.values
            results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
            logger.info('----------------------------------------------------------------------------------------')
            logger.info(f'@@@For:\tLoss: {loss}\tActivation: {activation}: Got results of {results}@@@')
            logger.info('----------------------------------------------------------------------------------------')


            print(f"###Completed full flow for {loss} and {activation}")
        except Exception as ex:
            print(f"^^^Failed full flow for {loss} and {activation}")


if __name__ == '__main__':
    # from keras.models import load_model
    # from evaluate.statistical import f1_score, recall_score, precision_score
    # model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180831_1715_15\\vqa_model_.h5'
    # model = load_model(model_location,
    #            custom_objects={'f1_score': f1_score,
    #                            'recall_score': recall_score,
    #                            'precision_score': precision_score})
    #
    # model_fn =model_location
    # VqaModelTrainer.model_2_db(model, model_fn, fn_history=None, notes='')
    main()
