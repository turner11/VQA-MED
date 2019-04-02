from functools import partial
from typing import Iterable, List, Any

from sqlalchemy import Column, Integer, String, Float, create_engine, ForeignKey, PrimaryKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker, relationship
import pandas as pd
from common.constatns import _DB_FILE_LOCATION

Base = declarative_base()

_engine = create_engine(f'sqlite:///{_DB_FILE_LOCATION}', echo=False)


class ModelScore(Base):
    __tablename__ = 'scores'
    # __table_args__ = (
    #     PrimaryKeyConstraint('model_id'),
    # )

    # id = Column('id', primary_key=True)
    model_id = Column('model_id', ForeignKey('models.id'), primary_key=True)
    bleu = Column('bleu', Float)
    wbss = Column('wbss', Float)
    models = relationship("Model", lazy='subquery', back_populates="model_scores")

    def __init__(self, model_id, bleu, wbss):
        """"""
        self.model_id = model_id
        self.bleu = bleu
        self.wbss = wbss

    def __repr__(self):
        return f'{self.__class__.__name__}(model_id={self.model_id}, bleu={self.bleu}, wbss={self.wbss})'

class QuestionCatgory(Base):
    """"""
    __tablename__ = 'question_categories'

    id = Column('id', Integer, primary_key=True)
    name = Column('name', String(50), primary_key=True)

    def __init__(self, name):
        """"""
        super().__init__()
        self.name = name


class EvaluationType(Base):
    """"""
    __tablename__ = 'evaluation_types'

    id = Column('id', Integer, primary_key=True)
    name = Column('name', String(50))

    def __init__(self, name):
        """"""
        super().__init__()
        self.name = name


class ModelPartialScore(Base):
    __tablename__ = 'partial_scores'
    # __table_args__ = (
    #     PrimaryKeyConstraint('model_id'),
    # )

    # id = Column('id', primary_key=True)
    model_id = Column('model_id', ForeignKey('models.id'), primary_key=True)
    evaluation_type = Column('evaluation_type', ForeignKey('evaluation_types.id'), primary_key=True)
    question_category_id = Column('question_category_id', ForeignKey('question_categories.id'), primary_key=True)
    score = Column('score', Float)



    def __init__(self, model_id, evaluation_type, question_category_id,score):
        """"""
        self.model_id = model_id
        self.evaluation_type = evaluation_type
        self.question_category_id = question_category_id
        self.score = score

    def __repr__(self):
        return f'{self.__class__.__name__}(evaluation_type={self.evaluation_type}, question_category_id={self.question_category_id},' \
            f' score={self.score})'


class Model(Base):
    __tablename__ = 'models'
    id = Column('id', Integer, primary_key=True)
    model_location = Column(String(50))
    history_location = Column(String(50))
    image_base_net = Column(String(50))
    loss = Column(Float())
    val_loss = Column(Float())
    accuracy = Column(Float())
    val_accuracy = Column(Float())
    class_strategy = Column(String(15))
    parameter_count = Column('parameter_count', Integer)
    trainable_parameter_count = Column('trainable_parameter_count', Integer)

    f1_score = Column(Float())
    f1_score_val = Column(Float())
    recall = Column(Float())
    recall_val = Column(Float())
    precsision = Column(Float())
    precsision_val = Column(Float())
    loss_function = Column(String(50))
    activation = Column(String(50))

    notes = Column('notes', String(200))
    # model_scores = relationship(ModelScore, backref='models')
    model_scores = relationship("ModelScore", lazy='subquery', back_populates="models")

    @property
    def score(self):
        assert len(self.model_scores) <= 1, f'Unexpectedly Got multiple scores for model {self.id}'
        s = next((m for m in self.model_scores), None)
        return s

    def __init__(self,
                 model_location,
                 history_location,
                 image_base_net,
                 loss,
                 val_loss,
                 accuracy,
                 val_accuracy,
                 notes,
                 parameter_count, trainable_parameter_count,
                 f1_score,
                 f1_score_val,
                 recall,
                 recall_val,
                 precsision,
                 precsision_val,
                 loss_function,
                 activation,
                 class_strategy):
        """"""
        self.model_location = model_location
        self.history_location = history_location
        self.image_base_net = image_base_net
        self.loss = loss
        self.val_loss = val_loss
        self.accuracy = accuracy
        self.val_accuracy = val_accuracy
        self.notes = notes
        self.parameter_count = parameter_count
        self.trainable_parameter_count = trainable_parameter_count
        self.f1_score = f1_score
        self.f1_score_val = f1_score_val
        self.recall = recall
        self.recall_val = recall_val
        self.precsision = precsision
        self.precsision_val = precsision_val
        self.loss_function = loss_function
        self.activation = activation
        self.class_strategy = class_strategy

    def __repr__(self):
        return f'{self.__class__.__name__}(id={self.id},\n' \
            f'\tmodel_location={self.model_location},\n' \
            f'\thistory_location={self.history_location},\n' \
            f'\timage_base_net={self.image_base_net},\n' \
            f'\tloss={self.loss},\n' \
            f'\tval_loss={self.val_loss},\n' \
            f'\taccuracy={self.accuracy},\n' \
            f'\tval_accuracy={self.val_accuracy},\n' \
            f'\tclass_strategy={self.class_strategy})' \
            f'\tf1_score = {self.f1_score},\n' \
            f'\tf1_score_val = {self.f1_score_val},\n' \
            f'\trecall = {self.recall},\n' \
            f'\trecall_val = {self.recall_val},\n' \
            f'\tprecsision = {self.precsision},\n' \
            f'\tprecsision_val = {self.precsision_val},\n' \
            f'\tloss_function = {self.loss_function},\n' \
            f'\tactivation = {self.activation},\n'.rstrip()


def create_db():
    Base.metadata.create_all(_engine)


def insert_dals(dal_obj_arr: Iterable[Base]) -> None:
    session = get_session()

    try:
        session.add_all(dal_obj_arr)
        # session.flush()
        session.commit()
    except Exception as ex:
        session.rollback()
        raise


def insert_dal(dal_obj: Base) -> None:
    return insert_dals([dal_obj])


def get_session():
    SessionMaker = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    session = SessionMaker()
    return session


def get_items(dal_type: Base) -> [Base]:
    session = get_session()
    res_q = session.query(dal_type)
    models = list(res_q)

    return models


get_models = partial(get_items, Model)
get_scores = partial(get_items, ModelScore)
get_partial_scores = partial(get_items, ModelPartialScore)


def get_model(predicate: callable) -> Model:
    models = get_models()
    return next(model for model in models if predicate(model))


def get_model_by_id(model_id: int) -> Model:
    return get_model(lambda model: model.id == model_id)


def get_models_data_frame():
    models = get_models()
    if not models:
        return pd.DataFrame()
    variables = [v for v in models[0].__dict__.keys() if not v.startswith('_')]
    df = pd.DataFrame([[getattr(i, j) for j in variables] for i in models], columns=variables)

    scores = get_scores()
    if len(scores) > 0:
        s_variables = [v for v in scores[0].__dict__.keys() if not v.startswith('_')] if scores else []
        s_df = pd.DataFrame([[getattr(i, j) for j in s_variables] for i in scores], columns=s_variables)


        merged_df = s_df.merge(df, left_on='model_id', right_on='id', how='outer')  # how='left')
        ret = merged_df
    else:
        # Should happen only before we have done ANY evaluations
        ret = df

    return ret


def execute_sql(txt):
    from sqlalchemy.sql import text
    with _engine.connect() as con:
        con.execute(text(txt))


def execute_sql_from_file(file_name):
    with open(file_name, 'r') as f:
        txt = f.read()
    return execute_sql(txt)


def main():
    # create_db()
    return
    # ps = ModelPartialScore(1,1,4,0.5)
    # insert_dal(ps)
    return
    # create_db()
    # return
    # # ms = get_models()
    # df_models = get_models_data_frame()
    #
    # return
    # resnet_model = Model(model_location='C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180730_0524_48\\vqa_model_ClassifyStrategies.CATEGORIAL_trained.h5',
    #                      history_location='',
    #                      image_base_net='resnet50',
    #                      loss=0.1248,
    #                      val_loss=2.7968,
    #                      accuracy=0.9570,
    #                      val_accuracy=0.5420,
    #                      notes ='Categorial, 4 options Imaging devices')

    # loss: - acc:  - val_loss: - val_acc:  Training Model: 3:50:30.246880
    vgg19_model_multi_classes = Model(
        model_location='C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180829_1113_47\\vqa_model_ClassifyStrategies.CATEGORIAL_trained.h5',
        history_location='',
        image_base_net='vgg19',
        loss=26.7192,
        val_loss=18.9696,
        accuracy=0.066,
        val_accuracy=0.0064,
        notes='',
        parameter_count=21061245,
        trainable_parameter_count=1036221,
        f1_score=0.0579,
        f1_score_val=0.0520,
        recall=0.0318,
        recall_val=0.0296,
        precsision=0.3887,
        precsision_val=0.2135,
        loss_function='categorical_crossentropy',
        activation='softmax'
    )

    # model_location = 'C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180829_1113_47\\vqa_model_ClassifyStrategies.CATEGORIAL_trained.h5',
    # history_location = '',
    # image_base_net = 'vgg19',
    # loss = 26.7192,
    # val_loss = 18.9696,
    # accuracy = 0.066,
    # val_accuracy = 0.0064,
    # notes = '',
    # parameter_count = 21061245,
    # trainable_parameter_count = 1036221,
    # f1_score = 0.0579,
    # f1_score_val = 0.0520,
    # recall = 0.0318,
    # recall_val = 0.0296,
    # precsision = 0.3887,
    # precsision_val = 0.2135,
    # loss_function = 'categorical_crossentropy',
    # activation = 'softmax',
    str()

    # vgg19_model = Model(
    #     model_location='C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180731_0630_29\\vqa_model_ClassifyStrategies.CATEGORIAL_trained.h5',
    #     history_location='',
    #     image_base_net='vgg19',
    #     loss=0.0843,
    #     val_loss=2.7968,
    #     accuracy=0.9776,
    #     val_accuracy=0.6480,
    #     notes='Categorial, 4 options Imaging devices')

    insert_dal(vgg19_model_multi_classes)
    # ## Resnet 50:
    # trained_model_location = 'C:\Users\Public\Documents\Data\2018\vqa_models\20180730_0524_48\vqa_model_ClassifyStrategies.CATEGORIAL_trained.h5'
    # loss: 0.1248 - acc: 0.9570 - val_loss: 2.7968 - val_acc: 0.5420
    # Training Model: 12:22:54.619203
    pass


if __name__ == '__main__':
    main()
