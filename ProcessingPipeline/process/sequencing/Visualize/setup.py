from distutils.core import setup

setup(name='interactive',
      version='0.1',
      packages=['interactive'],
      package_data={'interactive': ['*.coffee',
                                    'example_df.txt',
                                    'jan_ratios.csv',
                                    'template_inline.html',
                                    '*.json',
                                   ],
                   },
     )
