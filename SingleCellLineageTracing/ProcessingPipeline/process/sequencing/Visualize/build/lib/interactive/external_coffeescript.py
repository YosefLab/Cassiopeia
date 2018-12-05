import os
import bokeh.models

def load_code_template(key):
    fn = os.path.join(os.path.dirname(__file__), '{0}.coffee'.format(key))
    with open(fn) as fh:
        callback = fh.read()
    return callback

def build_callback(key, format_kwargs=None, args=None):
    if args is None:
        args = {}
    if format_kwargs is None:
        format_kwargs = {}

    code_template = load_code_template(key)
    code = code_template.format(**format_kwargs)
    name = '{0}_callback'.format(key)
    callback = bokeh.models.CustomJS.from_coffeescript(code=code, args=args)
    callback.name = name

    return callback
