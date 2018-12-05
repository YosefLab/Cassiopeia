def Annotation_factory(fields, read_only=False, **kwargs):
    if read_only:
        curlies = ['{' + name + ':' + spec[-1] + '}' for name, spec in fields]
    else:
        curlies = ['{' + name + ':' + spec.format(**kwargs) + '}'
                   for name, spec in fields
                  ]
    template = '_'.join(curlies).format
    names = [name for name, spec in fields]
    cast_function = {'s': str,
                     'd': int,
                     'f': float,
                    }
    casts = [cast_function[spec[-1]] for name, spec in fields]

    class Annotation(dict):
        @classmethod
        def from_identifier(cls, identifier):
            # Extracts an annotation from a string whose right-most underscore
            # separated fields are the values.
            values = identifier.rsplit('_', len(names) - 1)
            return cls({n: c(v) for n, c, v in zip(names, casts, values)})
        
        @classmethod
        def from_prefix_identifier(cls, identifier):
            # Extracts an annotation from a string whose left-most underscore
            # separated fields are the values.
            values = identifier.split('_', len(names))
            return cls({n: c(v) for n, c, v in zip(names, casts, values)})

        @classmethod
        def from_annotation(cls, annotation):
            return cls({name: annotation[name] for name in names})
        
        @property
        def identifier(self):
            return template(**self)

        def __str__(self):
            return template(**self)

    return Annotation

def make_convertor(encoded_as, convert_to):
    ''' Returns a function that takes a SAM line or AlignedSegment, interprets
        the QNAME field as an identifier for an encoded_as Annotation, and
        converts this to a convert_to Annotation.
    '''
    def converter(line):
        if isinstance(line, str): 
            qname = line.split('\t')[0]
        else:
            qname = line.qname
        encoded = encoded_as.from_identifier(qname)
        converted = convert_to.from_annotation(encoded)
        return converted

    return converter
