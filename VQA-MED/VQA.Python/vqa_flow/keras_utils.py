import logging
logger = logging.getLogger(__name__)
# def log_model_summary(model, print_fn):
#     # print_fn = lambda x: fh.write(x + '\n')
#     model.summary(print_fn=print_fn)

def print_model_summary_to_file(fn, model):
    # Open the file
    with open(fn,'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))