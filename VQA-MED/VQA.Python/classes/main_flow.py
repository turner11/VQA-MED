import itertools

from common.DAL import ModelScore
from vqa_logger import logger
from common.utils import VerboseTimer
from collections import namedtuple
ModelResults = namedtuple('ModelResults', ['loss', 'activation', 'bleu', 'wbss'])
from common import DAL




def main():
    ev = predict_test(126)
    print (ev)
    return
    evaluate_model(162)
    return
    train_model(model_id=85, optimizer='Adam', post_concat_dense_units=16)
    # evaluate_missing_models()
    # train_all()
    # add_scores()


def predict_test(model_id):
    # 162: 	WBSS: 0.143	BLEU 0.146

    from classes.vqa_model_predictor import DefaultVqaModelPredictor
    mp = DefaultVqaModelPredictor(model=model_id)
    validation_prediction = mp.predict(mp.df_test)
    predictions = validation_prediction.prediction.values

    strs = []
    for i, row in mp.df_test.iterrows():
        image = row["path"].rsplit('\\')[-1].rsplit('.',1)[0]
        s = f'{i+1}\t{image}\t{predictions[i]}'
        strs.append(s)

    res = '\n'.join(strs)
    return res

    str()




def evaluate_missing_models():
    from classes.vqa_model_predictor import DefaultVqaModelPredictor
    from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
    models = DAL.get_models()
    df_test, df_validation = None, None
    for model in models:
        try:
            if model.score:
                logger.debug(f'Model {model.id} has score: {model.score}')
            else:
                # if model.id == 70:
                #     continue
                logger.debug(f'Model {model.id} did not have a score')
                logger.debug('Loading predictor')
                mp = DefaultVqaModelPredictor(model=model, df_test=df_test, df_validation=df_validation )
                df_test, df_validation = mp.df_test, mp.df_validation

                logger.debug('predicting')
                validation_prediction = mp.predict(mp.df_validation)
                predictions = validation_prediction.prediction.values
                ground_truth = validation_prediction.answer.values
                logger.debug('evaluating')
                results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)

                ms = ModelScore(model_id=mp.model_idx_in_db, bleu=results['bleu'], wbss=results['wbss'])
                logger.debug(f'Created for {model.id}: {model.score}')
                logger.debug(f'inserting to db (model:{model.id})')
                DAL.insert_dal(ms)
        except Exception as ex:
            logger.error(f'Failed to evaluate model:{model.id}:\n{ex}')



def add_scores():
    mrs = \
        [
            ModelResults(loss='categorical_crossentropy', activation='softmax', bleu=0.2103365798666068,
                         wbss=0.1618619696290538),
            ModelResults(loss='categorical_crossentropy', activation='sigmoid', bleu=0.20716465254174818,
                         wbss=0.1626061924793508),
            ModelResults(loss='categorical_crossentropy', activation='relu', bleu=0.15736855849732678,
                         wbss=0.14544359117093514),
            ModelResults(loss='categorical_crossentropy', activation='tanh', bleu=0.12281738185824935,
                         wbss=0.13295906282738015),
            ModelResults(loss='binary_crossentropy', activation='softmax', bleu=0.20709800314167512,
                         wbss=0.16505984800946585),
            ModelResults(loss='binary_crossentropy', activation='sigmoid', bleu=0.011066186753122757,
                         wbss=0.07059635035866542),
            ModelResults(loss='binary_crossentropy', activation='relu', bleu=0.17719091533345543,
                         wbss=0.18922594334323695),
            ModelResults(loss='binary_crossentropy', activation='tanh', bleu=0.17134064846338432,
                         wbss=0.14325001923326888),
            ModelResults(loss='kullback_leibler_divergence', activation='softmax', bleu=0.21097555547161706,
                         wbss=0.1681282783438653),
            ModelResults(loss='kullback_leibler_divergence', activation='sigmoid', bleu=0.18821248683499342,
                         wbss=0.1554047580462376),
            ModelResults(loss='kullback_leibler_divergence', activation='relu', bleu=0.02697774568345777,
                         wbss=0.07583004566334839),
            ModelResults(loss='kullback_leibler_divergence', activation='tanh', bleu=0.09744046635625582,
                         wbss=0.1252390938198901),
            ModelResults(loss='poisson', activation='softmax', bleu=0.2113286732842069, wbss=0.16922309171600097),
            ModelResults(loss='poisson', activation='sigmoid', bleu=0.07885259511687942, wbss=0.09959880189229278),
            ModelResults(loss='poisson', activation='relu', bleu=0.18292855547920547, wbss=0.19228694357546577),
            ModelResults(loss='poisson', activation='tanh', bleu=0.17094432009760965, wbss=0.12700311138909282),
            ModelResults(loss='cosine_proximity', activation='softmax', bleu=0.17744901705310637,
                         wbss=0.14867902825006596),
            ModelResults(loss='cosine_proximity', activation='sigmoid', bleu=0.21354887627017066,
                         wbss=0.165517190063791),
            ModelResults(loss='cosine_proximity', activation='relu', bleu=0.21319867901229575,
                         wbss=0.16398778552485363),
            ModelResults(loss='cosine_proximity', activation='tanh', bleu=0.2131769238659314, wbss=0.16531721891857265),
            ModelResults(loss='mean_squared_error', activation='softmax', bleu=0.01705038450237624,
                         wbss=0.06610480116424522),
            ModelResults(loss='mean_squared_error', activation='sigmoid', bleu=0.03188850893171434,
                         wbss=0.06849953295109426),
            ModelResults(loss='mean_squared_error', activation='relu', bleu=0.10878383465433829,
                         wbss=0.1942925922238055),
            ModelResults(loss='mean_squared_error', activation='tanh', bleu=0.1208971391434066,
                         wbss=0.13415198668126332),
            ModelResults(loss='mean_absolute_error', activation='softmax', bleu=0.007377457835415172,
                         wbss=0.056711486212989555),
            ModelResults(loss='mean_absolute_error', activation='sigmoid', bleu=0.011066186753122757,
                         wbss=0.05060702607306806),
            ModelResults(loss='mean_absolute_error', activation='relu', bleu=0.15110741206227213,
                         wbss=0.3474417013103914),
            ModelResults(loss='mean_absolute_error', activation='tanh', bleu=0.008607034141317702,
                         wbss=0.06062920222217075),
            ModelResults(loss='mean_absolute_percentage_error', activation='softmax', bleu=0.20967127162659935,
                         wbss=0.16977735361334204),
            ModelResults(loss='mean_absolute_percentage_error', activation='sigmoid', bleu=0.004918305223610114,
                         wbss=0.055123040735561055),
            ModelResults(loss='mean_absolute_percentage_error', activation='relu', bleu=0.13164848029153245,
                         wbss=0.30421125399691823),
            ModelResults(loss='mean_absolute_percentage_error', activation='tanh', bleu=0.009836610447220229,
                         wbss=0.057061637779877626),
            ModelResults(loss='mean_squared_logarithmic_error', activation='softmax', bleu=0.008607034141317702,
                         wbss=0.06281686017414999),
            ModelResults(loss='mean_squared_logarithmic_error', activation='sigmoid', bleu=0.014754915670830341,
                         wbss=0.06720042790426754),
            ModelResults(loss='mean_squared_logarithmic_error', activation='relu', bleu=0.1096212108707012,
                         wbss=0.2306262082411956),
            ModelResults(loss='mean_squared_logarithmic_error', activation='tanh', bleu=0.08161636305331728,
                         wbss=0.10059043806662184),
            ModelResults(loss='squared_hinge', activation='softmax', bleu=0.009836610447220229,
                         wbss=0.059131262544423704),
            ModelResults(loss='squared_hinge', activation='sigmoid', bleu=0.19130433140548395, wbss=0.1555977820976305),
            ModelResults(loss='squared_hinge', activation='relu', bleu=0.13066012886469555, wbss=0.12980854372132594),
            ModelResults(loss='squared_hinge', activation='tanh', bleu=0.1548624908321035, wbss=0.14460742245693822),
            ModelResults(loss='hinge', activation='softmax', bleu=0.007377457835415172, wbss=0.06310635918402367),
            ModelResults(loss='hinge', activation='sigmoid', bleu=0.20888872792738816, wbss=0.16562621908978492),
            ModelResults(loss='hinge', activation='relu', bleu=0.14198420222651467, wbss=0.13949713675547962),
            ModelResults(loss='hinge', activation='tanh', bleu=0.17038926724670586, wbss=0.1477681303467244),
            ModelResults(loss='categorical_hinge', activation='softmax', bleu=0.14166722958709976,
                         wbss=0.13673830859283906),
            ModelResults(loss='categorical_hinge', activation='sigmoid', bleu=0.11804193279357406,
                         wbss=0.128798972499981),
            ModelResults(loss='categorical_hinge', activation='relu', bleu=0.1000009216111366,
                         wbss=0.11793839526803596),
            ModelResults(loss='categorical_hinge', activation='tanh', bleu=0.10783078390918409,
                         wbss=0.12366018074656748),
            ModelResults(loss='logcosh', activation='softmax', bleu=0.0012295763059025286, wbss=0.050907419909431734),
            ModelResults(loss='logcosh', activation='sigmoid', bleu=0.022750794548321275, wbss=0.07167066616004447),
            ModelResults(loss='logcosh', activation='relu', bleu=0.10784426570469208, wbss=0.22956577415180102),
            ModelResults(loss='logcosh', activation='tanh', bleu=0.10173751732258726, wbss=0.11651390737432664)
        ]

    from common.DAL import ModelScore
    score_dals = []
    for mr in mrs:
        model = DAL.get_model(lambda m: m.loss_function == mr.loss and m.activation == mr.activation)
        mid = model.id
        ms = ModelScore(mid, bleu=mr.bleu, wbss=mr.wbss)
        score_dals.append(ms)
    DAL.insert_dals(score_dals)

def train_model(model_id, optimizer, post_concat_dense_units=16):
    # Doing all of this here in order to not import tensor flow for other functions

    from classes.vqa_model_trainer import VqaModelTrainer
    from classes.vqa_model_builder import VqaModelBuilder
    from keras import backend as keras_backend
    keras_backend.clear_session()

    # Get------------------------------------------------------------------------
    model_dal = DAL.get_model_by_id(model_id=model_id)
    mb = VqaModelBuilder(model_dal.loss_function, model_dal.activation, post_concat_dense_units=post_concat_dense_units, optimizer=optimizer)
    model = mb.get_vqa_model()
    model_location, summary_fn, fn_image = VqaModelBuilder.save_model(model)

    # Train ------------------------------------------------------------------------

    batch_size = 75
    use_augmentation = True

    mt = VqaModelTrainer(model_location, use_augmentation=use_augmentation, batch_size=batch_size)
    history = mt.train()
    with VerboseTimer("Saving trained Model"):
        notes = f'post_concat_dense_units: {post_concat_dense_units};\n' \
                f'Optimizer: {optimizer}\n' \
                f'loss: {mb.loss_function}\n' \
                f'activation: {mb.dense_activation}\n' \
                f'epochs: {mt.epochs}\n' \
                f'batch_size: {batch_size}'
        logger.debug(f'Saving model')
        model_fn, summary_fn, fn_image, fn_history = VqaModelTrainer.save(mt.model, history, notes)
    logger.debug(f'Model saved to:\n\t{model_fn}')


    # Evaluate ------------------------------------------------------------------------
    results = evaluate_model()

    logger.info('----------------------------------------------------------------------------------------')
    logger.info(f'@@@For:\tLoss: {mb.loss_function}\tActivation: {mb.dense_activation}: Got results of {results}@@@')
    logger.info('----------------------------------------------------------------------------------------')


    print(f"###Completed full flow for {mb.loss_function} and {mb.dense_activation}")


def evaluate_model(model=None):
    from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
    from classes.vqa_model_predictor import DefaultVqaModelPredictor

    mp = DefaultVqaModelPredictor(model=model)
    validation_prediction = mp.predict(mp.df_validation)
    predictions = validation_prediction.prediction.values
    ground_truth = validation_prediction.answer.values
    results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
    ms = ModelScore(model_id=mp.model_idx_in_db, bleu=results['bleu'], wbss=results['wbss'])
    DAL.insert_dal(ms)
    return results



def train_all():
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


    optimizers = [ 'RMSprop', 'Adam']#['SGD', 'Adagrad', 'Adadelta', 'RMSprop', 'Adam']
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
                    ('mean_absolute_error', 'relu'),]

    la_units_opts = list(itertools.product(top_models, dense_units,optimizers))

    existing_scores = DAL.get_scores()
    models_ids = [s.model_id for s in existing_scores ]
    existing_models = DAL.get_models()
    models_with_scores = [m for m in existing_models if m.id in models_ids]

    # for loss, activation in losses_and_activations:
    for (loss, activation), post_concat_dense_units, opt in la_units_opts:
        try:

            def match(m):
                notes = (m.notes or '')
                is_curr_model =  m.loss_function == loss \
                       and m.activation == activation \
                       and opt in notes \
                       and str(post_concat_dense_units) in notes
                return is_curr_model





            match_model = next((m for m in models_with_scores if match(m)), None)
            if match_model is not None:
                print(f'Continuing for model:\n{match_model.notes}')
                continue



            # keras_backend.clear_session()


            mb = VqaModelBuilder(loss, activation,post_concat_dense_units=post_concat_dense_units, optimizer=opt)
            model = mb.get_vqa_model()
            model_fn, summary_fn, fn_image = VqaModelBuilder.save_model(model)

            # Train ------------------------------------------------------------------------

            # epochs=25
            # batch_size = 20
            keras_backend.clear_session()

            batch_size = 75
            use_augmentation = True



            model_location = model_fn

            mt = VqaModelTrainer(model_location, use_augmentation =use_augmentation , batch_size=batch_size)
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
    # from evaluate.VqaMedEvaluatorBase import VqaMedEvaluatorBase
    # from classes.vqa_model_builder import VqaModelBuilder
    # from classes.vqa_model_predictor import DefaultVqaModelPredictor
    # from classes.vqa_model_trainer import VqaModelTrainer
    # from keras import backend as keras_backend
    #
    # mp = DefaultVqaModelPredictor(model=None)
    #
    # validation_prediction = mp.predict(mp.df_validation)
    # predictions = validation_prediction.prediction.values
    # ground_truth = validation_prediction.answer.values
    # results = VqaMedEvaluatorBase.get_all_evaluation(predictions=predictions, ground_truth=ground_truth)
    #
    # loss, activation =('mean_absolute_error', 'relu')
    # ms = ModelScore(model_id=mp.model_idx_in_db, bleu=results['bleu'], wbss=results['wbss'])
    #
    #
    # DAL.insert_dal(ms)
    #
    # model_fn =model_location
    # VqaModelTrainer.model_2_db(model, model_fn, fn_history=None, notes='')





    main()
