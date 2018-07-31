from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
import pandas as pd
from common.constatns import _DB_FILE_LOCATION

Base = declarative_base()


_engine = create_engine(f'sqlite:///{_DB_FILE_LOCATION }', echo=False)
class Model(Base):
    __tablename__ = 'models'
    id = Column('id',Integer, primary_key=True)
    model_location = Column(String(50))
    history_location = Column(String(50))
    image_base_net = Column(String(50))
    loss = Column(Float())
    val_loss = Column(Float())
    accuracy = Column(Float())
    val_accuracy = Column(Float())
    class_strategy = Column(String(15))

    
    
    def __init__(self, model_location, history_location, image_base_net, loss, val_loss, accuracy, val_accuracy,notes):
        """"""
        self.model_location = model_location  
        self.history_location = history_location  
        self.image_base_net = image_base_net  
        self.loss = loss  
        self.val_loss = val_loss  
        self.accuracy = accuracy  
        self.val_accuracy = val_accuracy
        self.notes = notes

    def __repr__(self):
        return f'{self.__class__.__name__}(id={self.id},\n'\
                    f'\tmodel_location={self.model_location},\n' \
                    f'\thistory_location={self.history_location},\n'\
                    f'\timage_base_net={self.image_base_net},\n' \
                    f'\tloss={self.loss},\n' \
                    f'\tval_loss={self.val_loss},\n' \
                    f'\taccuracy={self.accuracy},\n' \
                    f'\tval_accuracy={self.val_accuracy},\n' \
                    f'\tclass_strategy={self.class_strategy})'

def create_db():
    Base.metadata.create_all(_engine)

def insert_models(models):
    session = get_session()

    try:
        session.add_all(models)
        # session.flush()
        session.commit()
    except:
        session.rollback()
        raise


def get_session():
    Session = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    session = Session()
    return session


def get_models():
    session = get_session()
    res_q = session.query(Model)
    models = list(res_q)

    return models

def get_model(id):
    models = get_models()
    return next(m for m in models if m.id == id)



def get_models_data_frame():
    models = get_models()
    if not models:
        return pd.DataFrame()
    variables = [v for v in models[0].__dict__.keys() if not v.startswith('_')]
    df = pd.DataFrame([[getattr(i, j) for j in variables] for i in models], columns=variables)
    return df


def main():
    # create_db()
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

    vgg19_model = Model(
        model_location='C:\\Users\\Public\\Documents\\Data\\2018\\vqa_models\\20180731_0630_29\\vqa_model_ClassifyStrategies.CATEGORIAL_trained.h5',
        history_location='',
        image_base_net='vgg19',
        loss=0.0843,
        val_loss=2.7968,
        accuracy=0.9776,
        val_accuracy=0.6480,
        notes='Categorial, 4 options Imaging devices')

    models = [vgg19_model]
    insert_models(models)
    # ## Resnet 50:
    # trained_model_location = 'C:\Users\Public\Documents\Data\2018\vqa_models\20180730_0524_48\vqa_model_ClassifyStrategies.CATEGORIAL_trained.h5'
    # loss: 0.1248 - acc: 0.9570 - val_loss: 2.7968 - val_acc: 0.5420
    # Training Model: 12:22:54.619203
    pass


if __name__ == '__main__':
    main()
