from pathlib import Path
from zensols.pybuild import SetupUtil

su = SetupUtil(
    setup_path=Path(__file__).parent.absolute(),
    name="zensols.nlp",
    package_names=['zensols', 'resources'],
    package_data={'': ['*.conf', '*.yml']},
    description='A utility library to assist in parsing natural language text.',
    user='plandes',
    project='nlparse',
    keywords=['tooling'],
    has_entry_points=False,
).setup()
