class InvalidArgumentException(Exception):
    def __init__(self, argument_name, message=None, **kwargs):
        """"""
        super().__init__(message)
        self.argument_name = argument_name
        self.has_argument = 'argument' in kwargs
        self.argument = kwargs.get('argument', None)

    def __repr__(self):
        return f'{self.__class__.__name__}(argument_name={self.argument_name})'

    def __str__(self):
        msg = f'Got an invalid argument: {self.argument_name}. '
        if self.has_argument:
            msg += f'Type: {type(self.argument)}. Argument: {str(self.argument)[:50]}'

        return msg.strip()


class NoDataException(Exception):
    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(args, kwargs)

    def __repr__(self):
        return super().__repr__()
