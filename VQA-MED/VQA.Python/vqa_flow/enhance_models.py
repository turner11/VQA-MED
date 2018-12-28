from hyperas.distributions import uniform
import itertools
from common.DAL import ModelScore
from vqa_logger import logger
from common.utils import VerboseTimer
from collections import namedtuple

ModelResults = namedtuple('ModelResults', ['loss', 'activation', 'bleu', 'wbss'])
from common import DAL


def main():
    train_all()


def train_all():
    raise Exception(f'Implement hyperas, e.g {uniform}')
    
    # Doing all of this here in order to not import tensor flow for other functions
    from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
    from classes.vqa_model_builder import VqaModelBuilder
    from classes.vqa_model_predictor import DefaultVqaModelPredictor
    from classes.vqa_model_trainer import VqaModelTrainer
    from keras import backend as keras_backend
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

    optimizers = ['RMSprop', 'Adam']  # ['SGD', 'Adagrad', 'Adadelta', 'RMSprop', 'Adam']
    dense_units = [16, 32]
    top_models = [
        ('cosine_proximity', 'sigmoid'),
        ('cosine_proximity', 'tanh'),
        ('cosine_proximity', 'relu'),
        ('poisson', 'softmax'),
        ('kullback_leibler_divergence', 'softmax'),
        ('mean_absolute_percentage_error', 'relu'),
        ('mean_squared_logarithmic_error', 'relu'),
        ('logcosh', 'relu'),
        ('mean_squared_error', 'relu'),
        ('mean_absolute_error', 'relu'), ]

    la_units_opts = list(itertools.product(top_models, dense_units, optimizers))

    existing_scores = DAL.get_scores()
    models_ids = [s.model_id for s in existing_scores]
    existing_models = DAL.get_models()
    models_with_scores = [m for m in existing_models if m.id in models_ids]

    # for loss, activation in losses_and_activations:
    for (loss, activation), post_concat_dense_units, opt in la_units_opts:
        try:

            def match(m):
                notes = (m.notes or '')
                is_curr_model = m.loss_function == loss \
                                and m.activation == activation \
                                and opt in notes \
                                and str(post_concat_dense_units) in notes
                return is_curr_model

            match_model = next((m for m in models_with_scores if match(m)), None)
            if match_model is not None:
                print(f'Continuing for model:\n{match_model.notes}')
                continue

            # keras_backend.clear_session()

            mb = VqaModelBuilder(loss, activation, post_concat_dense_units=post_concat_dense_units, optimizer=opt)
            model = mb.get_vqa_model()
            model_fn, summary_fn, fn_image = VqaModelBuilder.save_model(model)

            # Train ------------------------------------------------------------------------

            epochs = 1
            # batch_size = 20
            keras_backend.clear_session()

            batch_size = 75
            use_augmentation = True

            model_location = model_fn

            mt = VqaModelTrainer(model_location, use_augmentation=use_augmentation, batch_size=batch_size)
            history = mt.train()
            with VerboseTimer("Saving trained Model"):
                notes = f'post_concat_dense_units: {post_concat_dense_units};\n' \
                    f'Optimizer: {opt}\n' \
                    f'loss: {loss}\n' \
                    f'activation: {activation}\n' \
                    f'epochs: {epochs}\n' \
                    f'batch_size: {batch_size}'
                model_fn, summary_fn, fn_image, fn_history = VqaModelTrainer.save(mt.model, history, notes)
            print(model_fn)

            # Evaluate ------------------------------------------------------------------------
            mp = DefaultVqaModelPredictor(model=None)
            validation_prediction = mp.predict(mp.df_validation)
            predictions = validation_prediction.prediction.values
            ground_truth = validation_prediction.answer.values
            results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)

            ms = ModelScore(model_id=mp.model_idx_in_db, bleu=results['bleu'], wbss=results['wbss'])
            DAL.insert_dal(ms)
            logger.info('----------------------------------------------------------------------------------------')
            logger.info(f'@@@For:\tLoss: {loss}\tActivation: {activation}: Got results of {results}@@@')
            logger.info('----------------------------------------------------------------------------------------')

            print(f"###Completed full flow for {loss} and {activation}")
        except Exception as ex:
            import traceback as tb
            print(f"^^^Failed full flow for {loss} and {activation}\n:{ex}")
            tb.print_exc()
            tb.print_stack()


if __name__ == '__main__':
    main()
