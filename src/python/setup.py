from pathlib import Path
from zensols.pybuild import SetupUtil

su = SetupUtil(
    setup_path=Path(__file__).parent.absolute(),
    name="zensols.nlp",
    package_names=['zensols', 'resources'],
    # package_data={'': ['*.html', '*.js', '*.css', '*.map', '*.svg']},
    description='WRITE ME',
    user='plandes',
    project='nlparse',
    keywords=['tooling'],
    # has_entry_points=False,
).setup()
